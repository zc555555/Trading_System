"""
Daily prediction script for all 7 stocks.
Shows tomorrow's trading signals for AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA.
"""

import pickle
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from features import operators as ops
    from features.alphas_101_subset import compute_all_alphas, ALPHAS
except ImportError:
    import features.operators as ops
    from features.alphas_101_subset import compute_all_alphas, ALPHAS


def fetch_latest_data(symbols, days=300):
    """Fetch latest data for all symbols."""
    print(f"Fetching latest {days} days of data for {len(symbols)} stocks...")

    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df['symbol'] = symbol
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            all_data.append(df)
            print(f"  [OK] {symbol}: {len(df)} days")
        except Exception as e:
            print(f"  [FAIL] {symbol}: {e}")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(['date', 'symbol']).reset_index(drop=True)

    return df_all


def compute_features(df, config):
    """Compute all features."""
    print("\nComputing features...")

    # Compute alphas
    alpha_list = [f"alpha_{str(i).zfill(3)}" for i in config['features']['alphas_101']]
    available_alphas = [a for a in alpha_list if a in ALPHAS]

    df = compute_all_alphas(df, available_alphas)
    print(f"  Computed {len(available_alphas)} alphas")

    # Compute technical indicators
    tech_count = 0
    for tech in config['features'].get('technical', []):
        try:
            if tech == 'returns_1d':
                df[tech] = ops.returns(df, 'close', 1)
            elif tech == 'returns_5d':
                df[tech] = ops.returns(df, 'close', 5)
            elif tech == 'volatility_20d':
                df[tech] = ops.volatility(df, 20, 'close')
            elif tech == 'volume_ratio_20d':
                vol_ma = ops.ts_mean(df, 'volume', 20)
                df[tech] = df['volume'] / vol_ma
            elif tech == 'rsi_14':
                df[tech] = ops.rsi(df, 'close', 14)
            elif tech == 'macd':
                macd, signal, hist = ops.macd_indicator(df, 'close')
                df[tech] = macd
            tech_count += 1
        except Exception as e:
            print(f"  [FAIL] {tech}: {e}")

    print(f"  Computed {tech_count} technical indicators")

    return df


def main():
    print("=" * 80)
    print("TOMORROW'S PREDICTIONS FOR ALL 7 STOCKS")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    symbols = config['data']['symbols']

    # Load model
    artifacts_dir = Path(__file__).parent / "artifacts"
    model_path = artifacts_dir / "xgboost_model.pkl"

    if not model_path.exists():
        print("ERROR: Model not found!")
        print("Please run the training pipeline first:")
        print("  python run_build_fixed.py")
        print("  python train/train_xgb.py")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded: {model_path.name}\n")

    # Load feature manifest
    manifest_path = artifacts_dir / "feature_manifest.json"
    import json
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    feature_cols = manifest['feature_names']
    print(f"Features: {len(feature_cols)}")
    print(f"Feature list: {', '.join(feature_cols)}\n")

    # Fetch latest data
    df = fetch_latest_data(symbols, days=300)

    # Compute features
    df = compute_features(df, config)

    # Get latest row for each symbol
    latest_data = df.groupby('symbol').tail(1).reset_index(drop=True)

    # Make predictions
    predictions = []

    for idx, row in latest_data.iterrows():
        symbol = row['symbol']

        try:
            # Extract features
            features = row[feature_cols].values.reshape(1, -1)

            # Handle NaN
            features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values

            # Predict
            signal = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            predictions.append({
                'symbol': symbol,
                'date': row['date'].strftime('%Y-%m-%d'),
                'close': row['close'],
                'signal': signal,
                'prob_down': proba[0],
                'prob_up': proba[1],
                'confidence': max(proba)
            })
        except Exception as e:
            print(f"[FAIL] {symbol}: {e}")

    # Display results
    print("\n" + "=" * 80)
    print("TOMORROW'S TRADING SIGNALS")
    print("=" * 80)
    print()

    # Sort by probability (best opportunities first)
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('prob_up', ascending=False)

    # Display each stock
    for idx, pred in predictions_df.iterrows():
        symbol = pred['symbol']

        # Signal emoji
        if pred['signal'] == 1:
            signal_emoji = "BUY"
        else:
            signal_emoji = "HOLD/SELL"

        # Confidence level
        if pred['prob_up'] >= 0.60:
            confidence = "STRONG"
            action = "ENTER POSITION"
        elif pred['prob_up'] >= 0.55:
            confidence = "MEDIUM"
            action = "CONSIDER"
        elif pred['prob_up'] >= 0.50:
            confidence = "WEAK"
            action = "WATCH"
        else:
            confidence = "BEARISH"
            action = "AVOID"

        print(f"{symbol:6s} | ${pred['close']:7.2f} | {signal_emoji:10s} | "
              f"UP: {pred['prob_up']*100:5.1f}% | {confidence:8s} | {action}")

    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)
    print()

    # Buy recommendations (prob_up >= 55%)
    buy_candidates = predictions_df[predictions_df['prob_up'] >= 0.55].copy()

    if len(buy_candidates) > 0:
        print(f"BUY SIGNALS ({len(buy_candidates)} stocks):")
        for idx, pred in buy_candidates.iterrows():
            print(f"  {pred['symbol']:6s} - Probability UP: {pred['prob_up']*100:.1f}%")

        # If using $5000 portfolio
        print(f"\nSUGGESTED ALLOCATION (for $5000 portfolio):")
        total_weight = buy_candidates['prob_up'].sum()
        for idx, pred in buy_candidates.iterrows():
            weight = pred['prob_up'] / total_weight
            allocation = weight * 5000
            shares = int(allocation / pred['close'])
            print(f"  {pred['symbol']:6s}: ${allocation:7.2f} (~{shares} shares)")
    else:
        print("NO BUY SIGNALS")
        print("  Recommendation: Stay in cash today")

    print("\n" + "=" * 80)
    print("RISK WARNINGS")
    print("=" * 80)
    print("- This model has 52.57% win rate (slightly better than random)")
    print("- Past performance does not guarantee future results")
    print("- Use stop-loss: Exit if any position drops >2%")
    print("- Paper trade first before using real money")
    print("- Model trained on 2018-2024 data, may not work in 2026 market")
    print()


if __name__ == "__main__":
    main()
