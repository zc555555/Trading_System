"""
Fetch News and Analyze Sentiment using FinBERT.

Data sources:
1. News: Alpha Vantage (free tier: 25 requests/day)
2. Sentiment: FinBERT (pre-trained financial sentiment model)

For production: Use paid news APIs for more coverage.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import time
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# API Keys (you need to get these - free)
ALPHA_VANTAGE_KEY = "HT20821MMULTCZM8"  # Alpha Vantage API key


def get_news_alpha_vantage(symbol, api_key=ALPHA_VANTAGE_KEY):
    """
    Get news from Alpha Vantage.

    Free tier: 25 requests per day
    Returns: List of news articles with title, summary, time
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'feed' not in data:
            print(f"    [WARN] No news feed for {symbol}")
            return []

        articles = []
        for item in data['feed']:
            articles.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'time_published': item.get('time_published', ''),
                'source': item.get('source', ''),
                'url': item.get('url', '')
            })

        return articles

    except Exception as e:
        print(f"    [ERROR] {symbol}: {e}")
        return []


def analyze_sentiment_finbert(texts):
    """
    Analyze sentiment using FinBERT.

    FinBERT: Pre-trained BERT model fine-tuned on financial texts.
    Returns: Sentiment scores (positive, neutral, negative)
    """

    print("\n  Loading FinBERT model...")

    # Load FinBERT (this will download ~500MB on first run)
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()

        print("  FinBERT loaded successfully")

    except Exception as e:
        print(f"  [ERROR] Failed to load FinBERT: {e}")
        print("  Falling back to simple keyword-based sentiment")
        return simple_sentiment_analysis(texts)

    # Analyze each text
    sentiments = []

    for text in texts:
        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get probabilities
            # FinBERT outputs: [positive, negative, neutral]
            pos_prob = predictions[0][0].item()
            neg_prob = predictions[0][1].item()
            neu_prob = predictions[0][2].item()

            # Calculate sentiment score (-1 to +1)
            sentiment_score = pos_prob - neg_prob

            sentiments.append({
                'sentiment_score': sentiment_score,
                'positive_prob': pos_prob,
                'negative_prob': neg_prob,
                'neutral_prob': neu_prob
            })

        except Exception as e:
            print(f"    [WARN] Sentiment analysis failed: {e}")
            sentiments.append({
                'sentiment_score': 0.0,
                'positive_prob': 0.33,
                'negative_prob': 0.33,
                'neutral_prob': 0.34
            })

    return sentiments


def simple_sentiment_analysis(texts):
    """
    Fallback: Simple keyword-based sentiment analysis.
    """

    positive_words = ['buy', 'bullish', 'growth', 'profit', 'beat', 'exceed', 'strong', 'rise', 'gain', 'up', 'positive', 'optimistic']
    negative_words = ['sell', 'bearish', 'loss', 'miss', 'weak', 'fall', 'drop', 'down', 'negative', 'pessimistic', 'concern', 'risk']

    sentiments = []

    for text in texts:
        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total > 0:
            sentiment_score = (pos_count - neg_count) / total
        else:
            sentiment_score = 0.0

        sentiments.append({
            'sentiment_score': sentiment_score,
            'positive_prob': pos_count / max(total, 1),
            'negative_prob': neg_count / max(total, 1),
            'neutral_prob': 1 - (pos_count + neg_count) / max(total, 1)
        })

    return sentiments


def fetch_news_sentiment(symbols, api_key=ALPHA_VANTAGE_KEY):
    """
    Fetch news and analyze sentiment for multiple stocks.
    """

    print("=" * 80)
    print("FETCHING NEWS AND ANALYZING SENTIMENT")
    print("=" * 80)

    if api_key == "demo":
        print("\n[WARNING] Using demo API key (very limited)!")
        print("Get your free key at: https://www.alphavantage.co/support/#api-key")
        print("Free tier: 25 requests/day, 500 requests/month\n")

    all_news_data = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        # Fetch news
        articles = get_news_alpha_vantage(symbol, api_key)

        if len(articles) == 0:
            print(f"    No news articles found")
            continue

        print(f"    Found {len(articles)} articles")

        # Extract texts for sentiment analysis
        texts = []
        for article in articles:
            # Combine title and summary for better context
            text = f"{article['title']}. {article['summary']}"
            texts.append(text)

        # Analyze sentiment
        print(f"    Analyzing sentiment...")
        sentiments = analyze_sentiment_finbert(texts)

        # Combine news with sentiment
        for article, sentiment in zip(articles, sentiments):
            all_news_data.append({
                'symbol': symbol,
                'date': pd.to_datetime(article['time_published'][:8]),  # YYYYMMDD
                'title': article['title'],
                'summary': article['summary'],
                'source': article['source'],
                'sentiment_score': sentiment['sentiment_score'],
                'positive_prob': sentiment['positive_prob'],
                'negative_prob': sentiment['negative_prob'],
                'neutral_prob': sentiment['neutral_prob']
            })

        print(f"    Avg sentiment: {np.mean([s['sentiment_score'] for s in sentiments]):.3f}")

        # Rate limiting (Alpha Vantage: 5 requests/minute for free tier)
        if i < len(symbols):
            time.sleep(15)  # Wait 15 seconds between stocks

    # Create DataFrame
    print("\n" + "=" * 80)
    print("CREATING NEWS SENTIMENT DATASET")
    print("=" * 80)

    if len(all_news_data) == 0:
        print("\n[ERROR] No news data collected!")
        return None

    df_news = pd.DataFrame(all_news_data)

    print(f"\nNews sentiment dataset: {df_news.shape}")
    print(f"  Symbols: {df_news['symbol'].nunique()}")
    print(f"  Date range: {df_news['date'].min()} to {df_news['date'].max()}")
    print(f"  Total articles: {len(df_news):,}")

    # Aggregate by symbol and date (multiple articles per day)
    print("\nAggregating daily sentiment...")

    df_daily = df_news.groupby(['symbol', 'date']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'positive_prob': 'mean',
        'negative_prob': 'mean'
    }).reset_index()

    df_daily.columns = ['symbol', 'date', 'sentiment_mean', 'sentiment_std', 'news_count', 'positive_prob', 'negative_prob']

    print(f"Daily aggregated: {df_daily.shape}")

    # Save
    output_dir = Path(__file__).parent
    output_path = output_dir / "news_sentiment.parquet"

    df_daily.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_daily


def main():
    """Main execution."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Fetch news sentiment data')
    parser.add_argument('--top5', action='store_true', help='Use only top 5 stocks')
    args = parser.parse_args()

    # Load config (resolves named universe → symbols if needed)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import load_config
    config = load_config()
    symbols = config['data']['symbols']

    print(f"\nSymbols: {len(symbols)}")
    print("\nNote: With free API (25 req/day), this will take multiple days to complete.")
    print("Recommendation: Start with top 5 stocks or get paid API access.\n")

    # Option to use subset
    if args.top5:
        use_subset = 'y'
    else:
        use_subset = input("Use subset of top 5 stocks? (y/n): ").strip().lower()

    if use_subset == 'y':
        symbols = symbols[:5]
        print(f"Using top 5 stocks: {symbols}")

    # Fetch news sentiment
    df_news = fetch_news_sentiment(symbols)

    if df_news is not None:
        print("\n" + "=" * 80)
        print("NEWS SENTIMENT COLLECTION COMPLETE!")
        print("=" * 80)
        print("\nNext step: Process sentiment sequences with LSTM")
        print("  python features/process_news_with_lstm.py")


if __name__ == "__main__":
    main()
