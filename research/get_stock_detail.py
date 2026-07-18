"""
Get detailed stock data for web dashboard.
Returns historical prices and prediction.
"""

import pickle
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

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


def get_stock_detail(symbol, days=90):
    """Get detailed stock data including historical prices and prediction."""
    import warnings
    warnings.filterwarnings('ignore')

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load classification model (for buy/sell signal)
    artifacts_dir = Path(__file__).parent / "artifacts"
    model_path = artifacts_dir / "xgboost_model.pkl"
    with open(model_path, 'rb') as f:
        model_classification = pickle.load(f)

    # Load regression model (for price prediction)
    # Priority: Ensemble > XGBoost > None
    model_regression = None

    # Try ensemble model first (most accurate)
    ensemble_path = artifacts_dir / "ensemble_regression_model.pkl"
    if ensemble_path.exists():
        with open(ensemble_path, 'rb') as f:
            model_regression = pickle.load(f)
        print(f"Using ensemble model for predictions", file=sys.stderr)
    else:
        # Fall back to single XGBoost regression model
        regression_model_path = artifacts_dir / "xgboost_regression_model.pkl"
        if regression_model_path.exists():
            with open(regression_model_path, 'rb') as f:
                model_regression = pickle.load(f)
            print(f"Using XGBoost regression model", file=sys.stderr)
        else:
            print(f"No regression model found", file=sys.stderr)

    # Load feature manifest
    manifest_path = artifacts_dir / "feature_manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    feature_cols = manifest['feature_names']

    # Fetch historical data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days+300}d")  # Extra days for feature computation

    if len(df) == 0:
        return None

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df['symbol'] = symbol
    df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

    # Compute alphas (need all stocks for cross-sectional, so use cached version)
    alpha_list = [f"alpha_{str(i).zfill(3)}" for i in config['features']['alphas_101']]
    available_alphas = [a for a in alpha_list if a in ALPHAS]

    # For single stock, we'll compute alphas if possible
    if len(available_alphas) > 0:
        try:
            df = compute_all_alphas(df, available_alphas)
        except:
            pass

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
        except:
            pass

    # Get latest prediction
    latest_row = df.iloc[-1]

    try:
        features = latest_row[feature_cols].values.reshape(1, -1)
        features = pd.DataFrame(features, columns=feature_cols).ffill().fillna(0).values

        # Classification prediction (buy/sell signal)
        signal = int(model_classification.predict(features)[0])
        proba = model_classification.predict_proba(features)[0]

        # Regression prediction (exact price change)
        predicted_return = None
        predicted_price = None
        if model_regression is not None:
            predicted_return = float(model_regression.predict(features)[0])
            predicted_price = float(latest_row['close']) * (1 + predicted_return)

        prediction = {
            'symbol': symbol,
            'date': latest_row['date'].strftime('%Y-%m-%d'),
            'close': float(latest_row['close']),
            'signal': signal,
            'prob_down': float(proba[0]),
            'prob_up': float(proba[1]),
            'confidence': float(max(proba)),
            'predicted_return': predicted_return,  # NEW: exact return prediction
            'predicted_price': predicted_price      # NEW: tomorrow's predicted price
        }
    except:
        prediction = None

    # Get last N days of historical prices
    historical = df[['date', 'close']].tail(days).copy()
    historical['date'] = historical['date'].dt.strftime('%Y-%m-%d')
    historical['close'] = historical['close'].astype(float)

    # Generate future trend (use regression prediction if available)
    current_price = float(latest_row['close'])
    if prediction and prediction.get('predicted_price'):
        # Use regression model's prediction for accurate line
        future_trend = create_price_trend_line(current_price, prediction['predicted_price'])
    else:
        # Fallback to simulation
        volatility = float(latest_row.get('volatility_20d', 0.015))
        prob_up = float(prediction['prob_up']) if prediction else 0.5
        future_trend = simulate_future_trend(symbol, current_price, prob_up, volatility)

    result = {
        'symbol': symbol,
        'historical_prices': historical.to_dict('records'),
        'prediction': prediction,
        'current_price': float(latest_row['close']),
        'change_1d': float(latest_row.get('returns_1d', 0)) if 'returns_1d' in latest_row else 0,
        'volume': int(latest_row['volume']),
        'future_trend': future_trend  # NEW: Direction 5
    }

    return result


def create_price_trend_line(current_price, predicted_price):
    """
    Create a straight line from current price to predicted price.
    This represents the model's best estimate of tomorrow's price movement.
    """
    num_points = 50

    # Create time labels (9:30 AM to 4:00 PM)
    start_time = 9.5
    end_time = 16.0
    time_points = np.linspace(start_time, end_time, num_points)

    # Create straight line from current to predicted
    prices = np.linspace(current_price, predicted_price, num_points)

    def format_time(t):
        hours = int(t)
        minutes = int((t - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    trend_data = [
        {
            'time': format_time(t),
            'price': float(p)
        }
        for t, p in zip(time_points, prices)
    ]

    return trend_data


def simulate_future_trend(symbol, current_price, prob_up, volatility_20d):
    """
    Simulate next day's intraday price movement.
    Direction 5: Simple future trend visualization.
    """
    import numpy as np

    num_points = 50

    # Expected direction based on model prediction
    if prob_up > 0.5:
        drift = (prob_up - 0.5) * 0.02
    else:
        drift = -(0.5 - prob_up) * 0.02

    vol_scale = volatility_20d if not pd.isna(volatility_20d) else 0.015

    # Generate random walk with drift
    np.random.seed(hash(symbol) % 2**32)

    prices = [current_price]
    for i in range(num_points - 1):
        random_move = np.random.normal(0, vol_scale / np.sqrt(num_points))
        drift_move = drift / num_points
        mean_reversion = -0.1 * (prices[-1] - current_price * (1 + drift * i / num_points)) / current_price

        change = drift_move + random_move + mean_reversion
        next_price = prices[-1] * (1 + change)
        prices.append(next_price)

    # Create time labels
    start_time = 9.5
    end_time = 16.0
    time_points = np.linspace(start_time, end_time, num_points)

    def format_time(t):
        hours = int(t)
        minutes = int((t - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    trend_data = [
        {
            'time': format_time(t),
            'price': float(p)
        }
        for t, p in zip(time_points, prices)
    ]

    return trend_data


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Symbol required"}))
        sys.exit(1)

    symbol = sys.argv[1].upper()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    # Suppress output
    import os
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        result = get_stock_detail(symbol, days)

        # Restore stdout
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if result:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps({"error": "Failed to fetch data"}))
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(json.dumps({"error": str(e)}))
    finally:
        try:
            devnull.close()
        except:
            pass


if __name__ == "__main__":
    main()
