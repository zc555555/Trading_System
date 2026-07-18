"""
GET DAILY TRADING SIGNALS - Plan A

Simple version using existing data.
Run every morning to get trading signals.

Usage:
    python get_daily_signals.py
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def calculate_win_rates(ensemble_dict, df_all):
    """Calculate historical win rates for BUY and SELL signals from recent data."""
    try:
        models = ensemble_dict['models']
        weights = ensemble_dict['weights']
        feature_cols = ensemble_dict['feature_cols']

        # Use last 6 months of data (with labels) to calculate win rates
        df_recent = df_all[df_all['label'].notna()].copy()

        # Take last 500 rows (roughly 20 trading days * 24 stocks)
        df_recent = df_recent.tail(500)

        if len(df_recent) < 100:
            return None

        # Get predictions
        X = df_recent[feature_cols].values
        X = pd.DataFrame(X, columns=feature_cols).ffill().fillna(0).values

        predictions = {}
        for model_name in sorted(models.keys()):
            pred = models[model_name].predict(X)
            predictions[model_name] = pred

        pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
        weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
        ensemble_pred = pred_matrix @ weights_array

        # Get actual returns
        y_true = df_recent['future_return'].values

        # Calculate win rates
        buy_mask = ensemble_pred > 0
        sell_mask = ensemble_pred < 0

        buy_correct = ((ensemble_pred > 0) & (y_true > 0)).sum()
        buy_total = buy_mask.sum()
        buy_win_rate = buy_correct / buy_total if buy_total > 0 else 0

        sell_correct = ((ensemble_pred < 0) & (y_true < 0)).sum()
        sell_total = sell_mask.sum()
        sell_win_rate = sell_correct / sell_total if sell_total > 0 else 0

        overall_correct = ((ensemble_pred > 0) == (y_true > 0)).sum()
        overall_win_rate = overall_correct / len(y_true)

        return {
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'overall_win_rate': overall_win_rate,
            'buy_total': int(buy_total),
            'sell_total': int(sell_total),
            'sample_size': len(df_recent)
        }
    except Exception as e:
        print(f"[WARNING] Could not calculate win rates: {e}")
        return None


def main():
    print("\n" + "=" * 80)
    print("DAILY TRADING SIGNALS - PLAN A")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    artifacts_dir = Path(__file__).parent / "artifacts"
    data_dir = Path(__file__).parent / "data"

    # Load model
    model_path = artifacts_dir / "ensemble_time_windows.pkl"
    print(f"\nLoading model...")

    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # Load data (using latest available data)
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # Calculate historical win rates from recent data
    print(f"Calculating historical win rates from recent data...")
    win_rates = calculate_win_rates(ensemble_dict, df)
    if win_rates:
        print(f"[OK] Win rates calculated from last {win_rates['sample_size']} samples")
    else:
        print(f"[INFO] Win rates not available")

    # Get latest date's data
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()

    print(f"Using data from: {latest_date}")
    print(f"Stocks available: {len(latest_data)}")

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

    # Add to dataframe
    latest_data = latest_data.copy()
    latest_data['prediction'] = ensemble_pred
    latest_data['confidence'] = np.abs(ensemble_pred)

    # Plan A configuration (from testing)
    config = {
        'n_top': 10,
        'min_confidence': 0.005,  # 0.5%
        'min_stocks': 7
    }

    # Select stocks
    latest_data_sorted = latest_data.sort_values('confidence', ascending=False)
    top_n = latest_data_sorted.head(config['n_top'])
    selected = top_n[top_n['confidence'] >= config['min_confidence']]

    n_stocks = len(selected)

    # Display
    print("\n" + "=" * 80)
    print("TODAY'S SIGNALS")
    print("=" * 80)

    if n_stocks >= config['min_stocks']:
        print(f"\n[TRADE] {n_stocks} stocks recommended\n")

        # Show overall win rates if available
        if win_rates:
            print(f"Historical Win Rates (last {win_rates['sample_size']} predictions):")
            print(f"  Overall: {win_rates['overall_win_rate']*100:.1f}%")
            print(f"  BUY signals: {win_rates['buy_win_rate']*100:.1f}% ({win_rates['buy_total']} BUY predictions)")
            print(f"  SELL signals: {win_rates['sell_win_rate']*100:.1f}% ({win_rates['sell_total']} SELL predictions)")
            print()

        total_conf = selected['confidence'].sum()

        print(f"{'Rank':<6} {'Stock':<8} {'Direction':<10} {'Prediction':<12} {'Confidence':<12} {'Win Rate':<12} {'Position %':<10}")
        print("-" * 90)

        for i, (_, row) in enumerate(selected.iterrows(), 1):
            symbol = row['symbol']
            pred = row['prediction']
            conf = row['confidence']
            pos_pct = (conf / total_conf) * 100

            direction = "BUY" if pred > 0 else "SELL"

            # Get win rate for this direction
            if win_rates:
                win_rate = win_rates['buy_win_rate'] if pred > 0 else win_rates['sell_win_rate']
                win_rate_str = f"{win_rate*100:.1f}%"
            else:
                win_rate_str = "N/A"

            print(f"{i:<6} {symbol:<8} {direction:<10} {pred*100:>+6.2f}%     "
                  f"{conf*100:>6.2f}%      {win_rate_str:<12} {pos_pct:>6.1f}%")

        # Allocation example
        print("\n" + "-" * 70)
        print("ALLOCATION FOR £100 CAPITAL:")
        print("-" * 70)

        capital = 100

        print(f"\n{'Stock':<8} {'Amount (£)':<15} {'Action':<10}")
        print("-" * 35)

        for _, row in selected.iterrows():
            symbol = row['symbol']
            conf = row['confidence']
            pred = row['prediction']
            pos_pct = (conf / total_conf)
            amount = capital * pos_pct
            action = "BUY" if pred > 0 else "SELL"

            print(f"{symbol:<8} {amount:>8.2f}         {action:<10}")

        print("-" * 35)
        print(f"{'TOTAL':<8} {capital:>8.2f}\n")

        # Top 5 most confident with win rates
        print("Top 5 Most Confident:")
        for i, (_, row) in enumerate(selected.head(5).iterrows(), 1):
            direction = "UP" if row['prediction'] > 0 else "DOWN"
            if win_rates:
                win_rate = win_rates['buy_win_rate'] if row['prediction'] > 0 else win_rates['sell_win_rate']
                print(f"  {i}. {row['symbol']}: {row['confidence']*100:.2f}% confidence, "
                      f"{row['prediction']*100:+.2f}% expected {direction}, "
                      f"{win_rate*100:.1f}% win rate")
            else:
                print(f"  {i}. {row['symbol']}: {row['confidence']*100:.2f}% confidence, "
                      f"{row['prediction']*100:+.2f}% expected {direction}")

    else:
        print(f"\n[SKIP] Insufficient signals ({n_stocks} < {config['min_stocks']})")
        print("Recommendation: Hold cash today\n")

        # Show overall win rates even when skipping
        if win_rates:
            print(f"Historical Win Rates (last {win_rates['sample_size']} predictions):")
            print(f"  Overall: {win_rates['overall_win_rate']*100:.1f}%")
            print(f"  BUY signals: {win_rates['buy_win_rate']*100:.1f}%")
            print(f"  SELL signals: {win_rates['sell_win_rate']*100:.1f}%")
            print()

        if len(top_n) > 0:
            print("Near-threshold candidates:")
            for i, (_, row) in enumerate(top_n.head(5).iterrows(), 1):
                direction = "BUY" if row['prediction'] > 0 else "SELL"
                if win_rates:
                    win_rate = win_rates['buy_win_rate'] if row['prediction'] > 0 else win_rates['sell_win_rate']
                    print(f"  {i}. {row['symbol']}: {row['confidence']*100:.2f}%, "
                          f"{direction}, {row['prediction']*100:+.2f}% expected, "
                          f"{win_rate*100:.1f}% win rate")
                else:
                    print(f"  {i}. {row['symbol']}: {row['confidence']*100:.2f}%, "
                          f"{direction}, {row['prediction']*100:+.2f}% expected")

    # Risk reminders
    print("\n" + "=" * 80)
    print("RISK MANAGEMENT CHECKLIST")
    print("=" * 80)
    print("""
[  ] Set max daily loss at 3% (£3 for £100 capital)
[  ] Set stop-loss at 2-3% per stock
[  ] Plan to sell tomorrow (1-day hold)
[  ] Track all trades in spreadsheet
[  ] Review strategy after 3 consecutive losses
    """)

    # Save signals
    output_file = artifacts_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"

    signals = {
        'generated_at': datetime.now().isoformat(),
        'data_date': str(latest_date),
        'should_trade': n_stocks >= config['min_stocks'],
        'n_stocks': int(n_stocks),
        'stocks': [
            {
                'rank': i + 1,
                'symbol': row['symbol'],
                'prediction': float(row['prediction']),
                'confidence': float(row['confidence']),
                'position_pct': float((row['confidence'] / total_conf) * 100)
            }
            for i, (_, row) in enumerate(selected.iterrows())
        ] if n_stocks >= config['min_stocks'] else [],
        'config': config
    }

    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"Signals saved to: {output_file}")

    print("\n" + "=" * 80)
    print("[COMPLETE]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
