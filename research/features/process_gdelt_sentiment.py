"""
Process GDELT News with FinBERT Sentiment Analysis.

Input: Raw GDELT news articles (gdelt_news_raw.parquet)
Output: Daily aggregated sentiment features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm


def analyze_sentiment_finbert(titles, batch_size=32):
    """
    Analyze sentiment using FinBERT (batch processing for speed).

    Args:
        titles: List of article titles
        batch_size: Number of titles to process at once

    Returns:
        List of sentiment scores
    """

    print("\n  Loading FinBERT model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()

        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        print(f"  FinBERT loaded successfully (device: {device})")

    except Exception as e:
        print(f"  [ERROR] Failed to load FinBERT: {e}")
        print("  Install transformers: pip install transformers torch")
        return None

    # Batch processing
    sentiments = []

    print(f"  Analyzing {len(titles)} articles...")

    for i in tqdm(range(0, len(titles), batch_size), desc="  Processing batches"):
        batch_titles = titles[i:i+batch_size]

        try:
            # Tokenize batch
            inputs = tokenizer(
                batch_titles,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Extract scores for batch
            for j in range(len(batch_titles)):
                # FinBERT outputs: [positive, negative, neutral]
                pos_prob = predictions[j][0].item()
                neg_prob = predictions[j][1].item()
                neu_prob = predictions[j][2].item()

                # Calculate sentiment score (-1 to +1)
                sentiment_score = pos_prob - neg_prob

                sentiments.append({
                    'sentiment_score': sentiment_score,
                    'positive_prob': pos_prob,
                    'negative_prob': neg_prob,
                    'neutral_prob': neu_prob
                })

        except Exception as e:
            print(f"    [WARN] Batch {i//batch_size} failed: {e}")
            # Fill with neutral sentiment for failed batch
            for _ in range(len(batch_titles)):
                sentiments.append({
                    'sentiment_score': 0.0,
                    'positive_prob': 0.33,
                    'negative_prob': 0.33,
                    'neutral_prob': 0.34
                })

    return sentiments


def aggregate_daily_sentiment(df_news):
    """
    Aggregate article-level sentiment to daily sentiment per stock.

    Args:
        df_news: DataFrame with article-level sentiment

    Returns:
        DataFrame with daily aggregated sentiment
    """

    print("\nAggregating daily sentiment...")

    # Group by symbol and date
    df_daily = df_news.groupby(['symbol', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'positive_prob': 'mean',
        'negative_prob': 'mean'
    }).reset_index()

    df_daily.columns = [
        'symbol', 'date',
        'sentiment_mean', 'sentiment_std', 'news_count',
        'positive_prob', 'negative_prob'
    ]

    # Fill NaN std with 0 (when only 1 article per day)
    df_daily['sentiment_std'] = df_daily['sentiment_std'].fillna(0)

    print(f"Daily aggregated: {df_daily.shape}")

    return df_daily


def create_sentiment_sequences(df_daily, sequence_length=7):
    """
    Create time series features from daily sentiment.

    Features:
    1. news_sentiment_trend: Trend in sentiment (improving/worsening)
    2. news_sentiment_volatility: Volatility of sentiment
    3. recent_negative_spike: Sudden negative news (binary)
    4. news_sentiment_momentum: Sentiment momentum
    5. news_coverage_intensity: Sudden increase in news volume
    6. news_sentiment_divergence: High std indicates conflicting news
    """

    print("\nCreating sentiment sequences...")

    all_features = []

    # Group by symbol
    for symbol in df_daily['symbol'].unique():
        symbol_data = df_daily[df_daily['symbol'] == symbol].sort_values('date')

        if len(symbol_data) < sequence_length:
            print(f"  [SKIP] {symbol}: Insufficient data ({len(symbol_data)} days)")
            continue

        sentiment_values = symbol_data['sentiment_mean'].fillna(0).values
        sentiment_std_values = symbol_data['sentiment_std'].fillna(0).values
        news_count_values = symbol_data['news_count'].fillna(0).values
        dates = symbol_data['date'].values

        # Create sequences
        for i in range(sequence_length, len(sentiment_values)):
            # Get past N days of sentiment
            sent_seq = sentiment_values[i-sequence_length:i]
            std_seq = sentiment_std_values[i-sequence_length:i]
            count_seq = news_count_values[i-sequence_length:i]

            # Feature 1: Sentiment trend (linear regression slope)
            x = np.arange(sequence_length)
            if np.std(sent_seq) > 0:
                sentiment_trend = np.polyfit(x, sent_seq, 1)[0]  # Slope
            else:
                sentiment_trend = 0.0

            # Feature 2: Sentiment volatility (std of sentiment)
            sentiment_volatility = np.std(sent_seq)

            # Feature 3: Recent negative spike
            # Check if last 2 days have significantly negative sentiment
            recent_sentiment = np.mean(sent_seq[-2:])
            avg_sentiment = np.mean(sent_seq[:-2])
            if recent_sentiment < avg_sentiment - 0.3:  # Threshold
                negative_spike = 1
            else:
                negative_spike = 0

            # Feature 4: Sentiment momentum
            # Change in sentiment (recent avg vs earlier avg)
            recent_avg = np.mean(sent_seq[-3:])
            earlier_avg = np.mean(sent_seq[:3])
            sentiment_momentum = recent_avg - earlier_avg

            # Feature 5: News coverage intensity
            # Sudden increase in news volume
            recent_news_count = np.mean(count_seq[-2:])
            avg_news_count = np.mean(count_seq)
            if avg_news_count > 0:
                news_intensity = recent_news_count / avg_news_count
            else:
                news_intensity = 1.0

            # Feature 6: Sentiment divergence
            # High std indicates conflicting news
            sentiment_divergence = np.mean(std_seq)

            all_features.append({
                'date': dates[i],
                'symbol': symbol,
                'news_sentiment_trend': sentiment_trend,
                'news_sentiment_volatility': sentiment_volatility,
                'recent_negative_spike': negative_spike,
                'news_sentiment_momentum': sentiment_momentum,
                'news_coverage_intensity': news_intensity,
                'news_sentiment_divergence': sentiment_divergence
            })

    df_features = pd.DataFrame(all_features)

    print(f"Generated sentiment features: {df_features.shape}")

    return df_features


def main():
    """Process GDELT news sentiment."""

    print("=" * 80)
    print("PROCESSING GDELT NEWS SENTIMENT")
    print("=" * 80)

    # Load GDELT news
    data_dir = Path(__file__).parent.parent / "data"
    news_path = data_dir / "gdelt_news_raw.parquet"

    if not news_path.exists():
        print(f"\n[ERROR] GDELT news not found at {news_path}")
        print("Please run: python data/fetch_gdelt_news.py")
        return

    print(f"\nLoading GDELT news from: {news_path}")
    df_news = pd.read_parquet(news_path)

    print(f"News data shape: {df_news.shape}")
    print(f"Symbols: {df_news['symbol'].nunique()}")
    print(f"Date range: {df_news['date'].min()} to {df_news['date'].max()}")

    # Analyze sentiment with FinBERT
    print("\n" + "=" * 80)
    print("ANALYZING SENTIMENT WITH FINBERT")
    print("=" * 80)

    sentiments = analyze_sentiment_finbert(df_news['title'].tolist(), batch_size=32)

    if sentiments is None:
        print("\n[ERROR] Sentiment analysis failed!")
        return

    # Add sentiment to news data
    for i, sentiment in enumerate(sentiments):
        for key, value in sentiment.items():
            df_news.loc[i, key] = value

    # Aggregate to daily sentiment
    df_daily = aggregate_daily_sentiment(df_news)

    # Create time series features
    df_features = create_sentiment_sequences(df_daily, sequence_length=7)

    if len(df_features) == 0:
        print("\n[ERROR] No features generated!")
        return

    # Save features
    output_path = data_dir / "news_features.parquet"
    df_features.to_parquet(output_path, index=False)

    print(f"\n" + "=" * 80)
    print("NEWS SENTIMENT FEATURE GENERATION COMPLETE!")
    print("=" * 80)

    print(f"\nGenerated features saved to: {output_path}")
    print(f"Features: {list(df_features.columns)}")
    print(f"Total records: {len(df_features):,}")

    # Show statistics
    print(f"\nFeature statistics:")
    cols = ['news_sentiment_trend', 'news_sentiment_volatility', 'news_sentiment_momentum',
            'news_coverage_intensity', 'news_sentiment_divergence']
    print(df_features[cols].describe())

    print(f"\nRecent negative spikes: {df_features['recent_negative_spike'].sum()} occurrences")

    print(f"\nNext step: Train with external features")
    print("  python train/train_with_external_features.py")


if __name__ == "__main__":
    main()
