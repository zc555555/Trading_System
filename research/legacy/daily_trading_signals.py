"""
DAILY TRADING SIGNALS - Plan A

Run this script every morning to get today's trading signals.

Usage:
    python daily_trading_signals.py

Output:
    - List of stocks to buy today
    - Confidence levels
    - Position sizing suggestions
"""

import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf


def get_latest_data(symbols, lookback_days=100):
    """Fetch latest data for all symbols"""
    print("Fetching latest market data...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    all_data = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) > 0:
                hist = hist.reset_index()
                hist['symbol'] = symbol
                all_data.append(hist)
        except Exception as e:
            print(f"  [WARNING] Failed to fetch {symbol}: {e}")

    if not all_data:
        return None

    df = pd.concat(all_data, ignore_index=True)
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    return df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]


def calculate_features(df, feature_generator):
    """Calculate features for latest data"""
    # This is a simplified version
    # In production, you should use the exact feature calculation
    # from your training pipeline

    print("Calculating features...")

    # For now, we'll use a placeholder
    # You need to implement the exact feature calculation here
    return None


def generate_signals(config_path=None):
    """Generate trading signals for today"""

    print("\n" + "=" * 80)
    print("DAILY TRADING SIGNALS - PLAN A")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    artifacts_dir = Path(__file__).parent / "research" / "artifacts"
    if not artifacts_dir.exists():
        artifacts_dir = Path(__file__).parent / "artifacts"

    model_path = artifacts_dir / "ensemble_time_windows.pkl"

    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("Please train the model first!")
        return

    print(f"\nLoading model from: {model_path}")
    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    print(f"Models loaded: {list(models.keys())}")
    print(f"Features required: {len(feature_cols)}")

    # Stock universe
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'WMT', 'HD', 'DIS', 'NFLX',
        'CRM', 'ADBE', 'ORCL', 'AMD', 'INTC', 'CAT',
        'BA', 'UNH', 'JNJ', 'NKE'
    ]

    print(f"\nStock universe: {len(symbols)} stocks")

    # For demo: Use most recent data from training set
    # In production: Fetch real-time data and calculate features
    data_dir = Path(__file__).parent / "research" / "data"
    if not data_dir.exists():
        data_dir = Path(__file__).parent / "data"

    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # Get latest date
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()

    print(f"\n[NOTE] Using latest available data: {latest_date}")
    print(f"[NOTE] In production, fetch real-time data from your broker/API")

    # Prepare features
    X = latest_data[feature_cols].values
    X = pd.DataFrame(X, columns=feature_cols).ffill().fillna(0).values

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    for model_name in sorted(models.keys()):
        pred = models[model_name].predict(X)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    # Add predictions to data
    latest_data = latest_data.copy()
    latest_data['prediction'] = ensemble_pred
    latest_data['confidence'] = np.abs(ensemble_pred)

    # Apply Plan A rules
    config = {
        'n_top': 10,
        'min_confidence': 0.005,  # 0.5%
        'min_stocks': 7
    }

    # Select top stocks
    latest_data_sorted = latest_data.sort_values('confidence', ascending=False)
    top_stocks = latest_data_sorted.head(config['n_top'])
    top_stocks_filtered = top_stocks[top_stocks['confidence'] >= config['min_confidence']]

    n_stocks = len(top_stocks_filtered)

    # Display results
    print("\n" + "=" * 80)
    print("TODAY'S TRADING SIGNALS")
    print("=" * 80)

    if n_stocks >= config['min_stocks']:
        print(f"\n[TRADE] {n_stocks} stocks recommended")
        print("\nRecommended positions:")
        print(f"\n{'Rank':<6} {'Stock':<8} {'Prediction':<12} {'Confidence':<12} {'Suggested %':<12}")
        print("-" * 60)

        total_confidence = top_stocks_filtered['confidence'].sum()

        for i, (_, row) in enumerate(top_stocks_filtered.iterrows(), 1):
            symbol = row['symbol']
            pred = row['prediction']
            conf = row['confidence']

            # Position sizing based on confidence
            position_pct = (conf / total_confidence) * 100

            direction = "BUY" if pred > 0 else "SELL"
            print(f"{i:<6} {symbol:<8} {direction:>4} {pred*100:>+6.2f}%   "
                  f"{conf*100:>6.2f}%      {position_pct:>6.1f}%")

        print("\n" + "-" * 60)
        print(f"Total allocation: 100%")

        # Capital allocation example
        print("\n" + "=" * 80)
        print("EXAMPLE: ALLOCATION FOR £100")
        print("=" * 80)

        capital = 100
        print(f"\n{'Stock':<8} {'Amount':<12} {'Direction':<10}")
        print("-" * 35)

        for _, row in top_stocks_filtered.iterrows():
            symbol = row['symbol']
            conf = row['confidence']
            pred = row['prediction']
            position_pct = (conf / total_confidence)
            amount = capital * position_pct

            direction = "BUY" if pred > 0 else "SELL"
            print(f"{symbol:<8} £{amount:>6.2f}      {direction:<10}")

        print("-" * 35)
        print(f"{'TOTAL':<8} £{capital:>6.2f}")

        # Risk warning
        print("\n" + "=" * 80)
        print("RISK MANAGEMENT")
        print("=" * 80)

        print("\nRecommended controls:")
        print(f"  - Max daily loss: 3% of capital (£{capital * 0.03:.2f})")
        print(f"  - Stop loss per stock: 2-3%")
        print(f"  - Hold period: 1 day (sell tomorrow)")
        print(f"  - Review after 3 consecutive losses")

    else:
        print(f"\n[SKIP] Not enough high-confidence signals")
        print(f"Only {n_stocks} stocks meet criteria (need {config['min_stocks']})")
        print("\nRecommendation: Wait for tomorrow")

        if len(top_stocks) > 0:
            print("\nTop candidates (but below threshold):")
            for i, (_, row) in enumerate(top_stocks.head(5).iterrows(), 1):
                print(f"  {i}. {row['symbol']}: {row['confidence']*100:.2f}% confidence")

    # Save signals to file
    output_file = artifacts_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"
    import json

    signals_data = {
        'date': datetime.now().isoformat(),
        'data_date': str(latest_date),
        'should_trade': n_stocks >= config['min_stocks'],
        'n_stocks': int(n_stocks),
        'stocks': [
            {
                'symbol': row['symbol'],
                'prediction': float(row['prediction']),
                'confidence': float(row['confidence']),
                'position_pct': float((row['confidence'] / total_confidence) * 100) if n_stocks >= config['min_stocks'] else 0
            }
            for _, row in top_stocks_filtered.iterrows()
        ] if n_stocks >= config['min_stocks'] else []
    }

    with open(output_file, 'w') as f:
        json.dump(signals_data, f, indent=2)

    print(f"\n{'-' * 80}")
    print(f"Signals saved to: {output_file}")
    print(f"{'-' * 80}")

    print("\n" + "=" * 80)
    print("END OF SIGNALS")
    print("=" * 80)


if __name__ == "__main__":
    try:
        generate_signals()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
