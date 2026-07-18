"""
Process News Sentiment sequences with LSTM to generate features.

Input: Historical news sentiment time series (7 days)
Output: 4 sentiment-based features:
  - news_sentiment_trend: Trend in sentiment (improving/worsening)
  - news_sentiment_volatility: Volatility of sentiment
  - recent_negative_spike: Sudden negative news (binary)
  - news_momentum: Sentiment momentum
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def create_sentiment_sequences(news_data, sequence_length=7):
    """
    Create sequences from news sentiment time series.

    Args:
        news_data: DataFrame with columns [date, symbol, sentiment_mean, ...]
        sequence_length: Number of days to look back

    Returns:
        DataFrame with sentiment features for each (date, symbol)
    """

    print("Creating sentiment sequences...")

    all_features = []

    # Group by symbol
    for symbol in news_data['symbol'].unique():
        symbol_data = news_data[news_data['symbol'] == symbol].sort_values('date')

        if len(symbol_data) < sequence_length:
            print(f"  [SKIP] {symbol}: Insufficient data ({len(symbol_data)} days)")
            continue

        sentiment_values = symbol_data['sentiment_mean'].fillna(0).values
        sentiment_std_values = symbol_data['sentiment_std'].fillna(0).values
        news_count_values = symbol_data['news_count'].fillna(0).values
        dates = symbol_data['date'].values

        # Create sequences
        for i in range(sequence_length, len(sentiment_values)):
            # Get past 7 days of sentiment
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
    """Process news sentiment data and generate features."""

    print("=" * 80)
    print("PROCESSING NEWS SENTIMENT WITH LSTM")
    print("=" * 80)

    # Load news sentiment data
    data_dir = Path(__file__).parent.parent / "data"
    news_path = data_dir / "news_sentiment.parquet"

    if not news_path.exists():
        print(f"\n[ERROR] News sentiment data not found at {news_path}")
        print("Please run: python data/fetch_news_sentiment.py")
        return

    print(f"\nLoading news sentiment from: {news_path}")
    df_news = pd.read_parquet(news_path)

    print(f"News data shape: {df_news.shape}")
    print(f"Symbols: {df_news['symbol'].nunique()}")
    print(f"Date range: {df_news['date'].min()} to {df_news['date'].max()}")

    # Generate features
    df_features = create_sentiment_sequences(df_news, sequence_length=7)

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

    print(f"\nNext step: Merge with main dataset and retrain")
    print("  These features will be added to the existing features")


if __name__ == "__main__":
    main()
