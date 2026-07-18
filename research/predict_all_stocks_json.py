"""
JSON output version for Go web server.
"""

import pickle
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

try:
    from features import operators as ops
    from features.alphas_101_subset import compute_all_alphas, ALPHAS
except ImportError:
    import features.operators as ops
    from features.alphas_101_subset import compute_all_alphas, ALPHAS


def fetch_latest_data(symbols, days=300):
    """Fetch latest data for all symbols."""
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
        except Exception:
            pass

    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values(['date', 'symbol']).reset_index(drop=True)
    return df_all


def compute_features(df, config):
    """Compute all features."""
    # Compute alphas
    alpha_list = [f"alpha_{str(i).zfill(3)}" for i in config['features']['alphas_101']]
    available_alphas = [a for a in alpha_list if a in ALPHAS]
    df = compute_all_alphas(df, available_alphas)

    # Compute technical indicators
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
            elif tech == 'macd':
                macd, signal, hist = ops.macd_indicator(df, 'close')
                df[tech] = macd
        except Exception:
            pass

    return df


def main():
    # Suppress all stderr and stdout except final JSON output
    import os
    import warnings
    warnings.filterwarnings('ignore')

    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        symbols = config['data']['symbols']

        # Load model
        artifacts_dir = Path(__file__).parent / "artifacts"
        model_path = artifacts_dir / "xgboost_model.pkl"

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load feature manifest
        manifest_path = artifacts_dir / "feature_manifest.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        feature_cols = manifest['feature_names']

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
                features = pd.DataFrame(features, columns=feature_cols).ffill().fillna(0).values

                # Predict
                signal = int(model.predict(features)[0])
                proba = model.predict_proba(features)[0]

                predictions.append({
                    'symbol': symbol,
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'signal': signal,
                    'prob_down': float(proba[0]),
                    'prob_up': float(proba[1]),
                    'confidence': float(max(proba))
                })
            except Exception:
                pass

        # Restore stdout for JSON output
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Output JSON
        print(json.dumps(predictions, indent=2))
    finally:
        # Cleanup
        try:
            devnull.close()
        except:
            pass


if __name__ == "__main__":
    # Suppress warnings for cleaner JSON output
    import warnings
    warnings.filterwarnings('ignore')

    main()
