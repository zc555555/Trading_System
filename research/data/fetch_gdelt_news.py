"""
Fetch News from GDELT (Global Database of Events, Language, and Tone).

GDELT provides:
- Free access to global news (2015-present)
- Built-in tone/sentiment scores
- Massive coverage (100+ languages, 1000+ sources)

Strategy:
- Phase 1: Test with 1 month of data for 5 stocks
- Phase 2: Expand to full historical range if effective

Data source: GDELT 2.0 GKG (Global Knowledge Graph)
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import time
import argparse
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')


# Company name mappings (for disambiguation)
COMPANY_NAMES = {
    'AAPL': ['Apple Inc', 'Apple Computer', 'Apple Corp'],
    'MSFT': ['Microsoft Corp', 'Microsoft Corporation'],
    'GOOGL': ['Google', 'Alphabet Inc', 'Alphabet Corp'],
    'AMZN': ['Amazon.com', 'Amazon Inc', 'Amazon Corp'],
    'NVDA': ['NVIDIA Corp', 'Nvidia Corporation'],
    'TSLA': ['Tesla Inc', 'Tesla Motors'],
    'META': ['Meta Platforms', 'Facebook Inc', 'Facebook'],
    'BRK.B': ['Berkshire Hathaway'],
    'UNH': ['UnitedHealth Group', 'United Health'],
    'JNJ': ['Johnson & Johnson', 'J&J'],
    'V': ['Visa Inc', 'Visa'],
    'XOM': ['Exxon Mobil', 'ExxonMobil'],
    'WMT': ['Walmart', 'Wal-Mart'],
    'JPM': ['JPMorgan Chase', 'JP Morgan'],
    'MA': ['Mastercard'],
    'PG': ['Procter & Gamble', 'P&G'],
    'LLY': ['Eli Lilly'],
    'CVX': ['Chevron Corp'],
    'ABBV': ['AbbVie Inc'],
    'HD': ['Home Depot'],
    'MRK': ['Merck & Co'],
    'KO': ['Coca-Cola', 'Coca Cola'],
    'PEP': ['PepsiCo'],
    'COST': ['Costco']
}


def fetch_gdelt_doc_api(query, start_date, end_date, max_records=250):
    """
    Fetch news from GDELT DOC 2.0 API.

    API Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
    Free tier: 250 records per query

    Args:
        query: Search query (company name)
        start_date: Start date (YYYYMMDDHHMMSS format)
        end_date: End date (YYYYMMDDHHMMSS format)
        max_records: Max records to fetch

    Returns:
        List of articles with metadata
    """

    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Format dates for GDELT API (YYYYMMDDHHMMSS)
    if isinstance(start_date, str) and len(start_date) == 10:  # YYYY-MM-DD
        start_date = start_date.replace('-', '') + '000000'
    if isinstance(end_date, str) and len(end_date) == 10:
        end_date = end_date.replace('-', '') + '235959'

    params = {
        'query': query,
        'mode': 'artlist',
        'maxrecords': max_records,
        'startdatetime': start_date,
        'enddatetime': end_date,
        'format': 'json'
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"    [ERROR] API returned status {response.status_code}: {response.text[:200]}")
            return []

        # Debug: print response
        if len(response.text) < 100:
            print(f"    [DEBUG] Response: {response.text}")

        data = response.json()

        if 'articles' not in data:
            print(f"    [WARN] No 'articles' key in response. Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            return []

        articles = []
        for idx, article in enumerate(data['articles']):
            seendate = article.get('seendate', '')
            # Debug: print first article's date format
            if idx == 0:
                print(f"    [DEBUG] First article seendate: '{seendate}' (type: {type(seendate)})")

            articles.append({
                'url': article.get('url', ''),
                'title': article.get('title', ''),
                'seendate': seendate,  # YYYYMMDDHHMMSS
                'socialimage': article.get('socialimage', ''),
                'domain': article.get('domain', ''),
                'language': article.get('language', 'eng'),
                'sourcecountry': article.get('sourcecountry', '')
            })

        return articles

    except Exception as e:
        print(f"    [ERROR] {e}")
        return []


def parse_gdelt_date(date_str):
    """Parse GDELT date format (20251201T170000Z) to datetime."""
    try:
        # GDELT uses ISO 8601 format: YYYYMMDDTHHMMSSZ
        return pd.to_datetime(date_str, format='%Y%m%dT%H%M%SZ')
    except:
        # Fallback: try standard format
        try:
            return pd.to_datetime(date_str, format='%Y%m%d%H%M%S')
        except:
            return None


def filter_relevant_articles(articles, symbol, company_names):
    """
    Filter articles that are actually about the company (disambiguation).

    Simple heuristic:
    - Title contains company name or stock symbol
    - Exclude articles about unrelated topics
    """

    filtered = []

    for article in articles:
        title = article['title'].lower()

        # Check if title contains symbol or company name
        is_relevant = False

        # Check symbol (with word boundaries)
        if f' {symbol.lower()} ' in f' {title} ' or f'${symbol.lower()}' in title:
            is_relevant = True

        # Check company names
        for name in company_names:
            if name.lower() in title:
                is_relevant = True
                break

        # Exclude obvious false positives for common words
        if symbol == 'V' and 'visa' not in title:  # Avoid "vs" matching Visa
            is_relevant = False
        if symbol == 'MA' and 'mastercard' not in title:  # Avoid "may" matching Mastercard
            is_relevant = False

        if is_relevant:
            filtered.append(article)

    return filtered


def deduplicate_articles(articles):
    """Remove duplicate articles based on URL and title similarity."""

    if len(articles) == 0:
        return []

    # Convert to DataFrame for easier deduplication
    df = pd.DataFrame(articles)

    # Remove exact URL duplicates
    df = df.drop_duplicates(subset=['url'], keep='first')

    # Simple title-based deduplication (exact match)
    df = df.drop_duplicates(subset=['title'], keep='first')

    return df.to_dict('records')


def fetch_gdelt_for_stocks(symbols, start_date, end_date, test_mode=True):
    """
    Fetch GDELT news for multiple stocks.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        test_mode: If True, only fetch 5 stocks for 1 month
    """

    print("=" * 80)
    print("FETCHING NEWS FROM GDELT")
    print("=" * 80)

    print(f"\nGDELT: Global Database of Events, Language, and Tone")
    print(f"Coverage: 2015-present, 100+ languages, 1000+ sources")
    print(f"Data: Free and unlimited")

    if test_mode:
        symbols = symbols[:5]
        print(f"\n[TEST MODE] Using {len(symbols)} stocks: {symbols}")

        # Use December 2025 for testing (more likely to be indexed)
        end_date = '2025-12-31'
        start_date = '2025-12-01'
        print(f"Date range: {start_date} to {end_date} (1 month)")
    else:
        print(f"\nSymbols: {len(symbols)}")
        print(f"Date range: {start_date} to {end_date}")

    all_news_data = []

    # Initial delay to respect rate limit
    print("\nWaiting 6 seconds to respect GDELT rate limit...")
    time.sleep(6)

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        company_names = COMPANY_NAMES.get(symbol, [symbol])
        print(f"    Company names: {company_names}")

        # Build query (search for any of the company names)
        # GDELT requires OR queries to be wrapped in parentheses
        query = '(' + ' OR '.join([f'"{name}"' for name in company_names]) + ')'

        # Fetch articles
        print(f"    Fetching articles from GDELT...")
        articles = fetch_gdelt_doc_api(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_records=250
        )

        print(f"    Found {len(articles)} articles")

        if len(articles) == 0:
            continue

        # Filter relevant articles (disambiguation)
        print(f"    Filtering relevant articles...")
        filtered = filter_relevant_articles(articles, symbol, company_names)
        print(f"    After filtering: {len(filtered)} relevant articles")

        # Deduplicate
        print(f"    Deduplicating...")
        deduplicated = deduplicate_articles(filtered)
        print(f"    After deduplication: {len(deduplicated)} unique articles")

        # Add symbol to each article
        for article in deduplicated:
            article['symbol'] = symbol
            article['date'] = parse_gdelt_date(article['seendate'])

        all_news_data.extend(deduplicated)

        # Rate limiting (GDELT requires 5+ seconds between requests)
        if i < len(symbols):
            time.sleep(6)  # 6 second delay between stocks

    # Create DataFrame
    print("\n" + "=" * 80)
    print("CREATING NEWS DATASET")
    print("=" * 80)

    if len(all_news_data) == 0:
        print("\n[ERROR] No news data collected!")
        return None

    df_news = pd.DataFrame(all_news_data)

    # Remove articles with invalid dates
    df_news = df_news[df_news['date'].notna()].copy()

    # Convert date to date only (no time)
    df_news['date'] = pd.to_datetime(df_news['date']).dt.date
    df_news['date'] = pd.to_datetime(df_news['date'])

    print(f"\nNews dataset: {df_news.shape}")
    print(f"  Symbols: {df_news['symbol'].nunique()}")
    print(f"  Date range: {df_news['date'].min()} to {df_news['date'].max()}")
    print(f"  Total articles: {len(df_news):,}")

    # Show distribution
    print(f"\nArticles per stock:")
    print(df_news['symbol'].value_counts())

    # Save raw articles
    output_dir = Path(__file__).parent
    output_path = output_dir / "gdelt_news_raw.parquet"

    df_news.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_news


def main():
    """Main execution."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Fetch news from GDELT')
    parser.add_argument('--full', action='store_true', help='Fetch full historical data (not recommended for first run)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default='2018-01-01')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default=datetime.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    # Load config (resolves named universe → symbols if needed)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import load_config
    config = load_config()
    symbols = config['data']['symbols']

    test_mode = not args.full

    if test_mode:
        print("\n[TEST MODE] Running with 5 stocks for 1 month")
        print("This will help validate the data quality before full download.")
        print("To fetch full data, use: python fetch_gdelt_news.py --full")
        print()

    # Fetch news
    df_news = fetch_gdelt_for_stocks(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        test_mode=test_mode
    )

    if df_news is not None:
        print("\n" + "=" * 80)
        print("GDELT NEWS COLLECTION COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Analyze sentiment with FinBERT:")
        print("     python features/process_gdelt_sentiment.py")
        print("  2. If test results look good, fetch full data:")
        print("     python data/fetch_gdelt_news.py --full")


if __name__ == "__main__":
    main()
