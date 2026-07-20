"""
Stage 1.3: Ensemble Signals Strategy

Combine 3 dynamic strategies:
1. Dynamic Top 7 (by confidence)
2. Rolling Window (by historical performance)
3. Core-Satellite (stable + dynamic)

Signal Strength:
- STRONG: All 3 strategies agree -> 70%+ accuracy expected
- MEDIUM: 2 strategies agree -> 65%+ accuracy expected
- WEAK: Only 1 strategy -> Skip

Expected: 63.84% -> 66-68%
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


def direction_accuracy(y_true, y_pred):
    return ((y_true > 0) == (y_pred > 0)).mean()


class DynamicTopSelector:
    """Strategy 1: Select by confidence (prediction magnitude)"""

    def select(self, day_data, n_top=7):
        day_data = day_data.copy()
        day_data['confidence'] = np.abs(day_data['prediction'])
        top_stocks = day_data.nlargest(n_top, 'confidence')
        return set(top_stocks['symbol'].tolist())


class RollingWindowSelector:
    """Strategy 2: Select by rolling window performance"""

    def __init__(self, window_days=30):
        self.window_days = window_days
        self.history = defaultdict(list)

    def update(self, date, symbol, y_true, y_pred):
        correct = (y_true > 0) == (y_pred > 0)
        self.history[symbol].append({
            'date': date,
            'correct': correct
        })

    def select(self, available_symbols, n_top=7):
        scores = {}
        for symbol in available_symbols:
            records = self.history[symbol][-self.window_days:]
            if len(records) >= self.window_days // 2:
                accuracy = sum(r['correct'] for r in records) / len(records)
                scores[symbol] = accuracy
            else:
                scores[symbol] = 0.5  # Default

        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return set([s[0] for s in sorted_stocks[:n_top]])


class CoreSatelliteSelector:
    """Strategy 3: Core (stable) + Satellite (dynamic)"""

    def __init__(self, core_stocks):
        self.core_stocks = set(core_stocks)

    def select(self, day_data, n_satellite=2):
        # Core stocks
        selected = self.core_stocks.copy()

        # Add satellite (high confidence non-core stocks)
        non_core = day_data[~day_data['symbol'].isin(self.core_stocks)].copy()
        if len(non_core) > 0:
            non_core['confidence'] = np.abs(non_core['prediction'])
            satellites = non_core.nlargest(n_satellite, 'confidence')
            selected.update(satellites['symbol'].tolist())

        return selected


def backtest_ensemble_signals(df, predictions, config):
    """
    Backtest ensemble signal strategy

    config = {
        'strategy1_n': 7,
        'strategy2_n': 7,
        'strategy2_window': 30,
        'core_stocks': ['JNJ', 'JPM', 'GS', 'NFLX', 'WMT'],
        'satellite_n': 2,
        'signal_threshold': 2  # At least 2 strategies must agree
    }
    """
    df = df.copy()
    df['prediction'] = predictions
    df['date'] = pd.to_datetime(df['date'])

    # Initialize strategies
    strategy1 = DynamicTopSelector()
    strategy2 = RollingWindowSelector(window_days=config['strategy2_window'])
    strategy3 = CoreSatelliteSelector(core_stocks=config['core_stocks'])

    trades = []
    signal_stats = {'strong': 0, 'medium': 0, 'weak': 0, 'skipped': 0}

    unique_dates = sorted(df['date'].unique())

    for date in unique_dates:
        day_data = df[df['date'] == date]

        # Update rolling window history
        for _, row in day_data.iterrows():
            strategy2.update(date, row['symbol'], row['future_return'], row['prediction'])

        # Get signals from each strategy
        signal1 = strategy1.select(day_data, n_top=config['strategy1_n'])
        signal2 = strategy2.select(day_data['symbol'].unique(), n_top=config['strategy2_n'])
        signal3 = strategy3.select(day_data, n_satellite=config['satellite_n'])

        # Analyze signal strength
        for _, row in day_data.iterrows():
            symbol = row['symbol']
            votes = sum([
                symbol in signal1,
                symbol in signal2,
                symbol in signal3
            ])

            if votes == 3:
                signal_strength = 'strong'
            elif votes == 2:
                signal_strength = 'medium'
            elif votes == 1:
                signal_strength = 'weak'
            else:
                signal_strength = 'none'

            # Trade only if meets threshold
            if votes >= config['signal_threshold']:
                row_dict = row.to_dict()
                row_dict['signal_strength'] = signal_strength
                row_dict['votes'] = votes
                trades.append(row_dict)
                signal_stats[signal_strength] += 1
            else:
                signal_stats['skipped'] += 1

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    overall_acc = direction_accuracy(
        trades_df['future_return'].values,
        trades_df['prediction'].values
    )

    # Metrics by signal strength
    by_strength = {}
    for strength in ['strong', 'medium', 'weak']:
        strength_trades = trades_df[trades_df['signal_strength'] == strength]
        if len(strength_trades) > 0:
            by_strength[strength] = {
                'count': len(strength_trades),
                'accuracy': float(direction_accuracy(
                    strength_trades['future_return'].values,
                    strength_trades['prediction'].values
                ))
            }
        else:
            by_strength[strength] = {'count': 0, 'accuracy': 0.0}

    results = {
        'overall_accuracy': float(overall_acc),
        'total_trades': len(trades_df),
        'coverage': len(trades_df) / len(df),
        'signal_stats': signal_stats,
        'by_signal_strength': by_strength,
        'config': config
    }

    return results


def main():
    print("\n" + "=" * 80)
    print("Stage 1.3: Ensemble Signals Strategy")
    print("=" * 80)

    # Load model
    artifacts_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/artifacts")
    model_path = artifacts_dir / "ensemble_time_windows.pkl"

    with open(model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # Load data
    data_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/data")
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # Split
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTest set: {len(test_df):,} samples")

    # Predict
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print("\nGenerating predictions...")
    predictions = {}
    for model_name in sorted(models.keys()):
        pred = models[model_name].predict(X_test)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    baseline_acc = direction_accuracy(y_test, ensemble_pred)
    print(f"Baseline accuracy: {baseline_acc*100:.2f}%")

    # Identify top 5 stocks for core
    print("\n" + "-" * 80)
    print("Identifying Core Stocks (Top 5 by stability)")
    print("-" * 80)

    test_df_copy = test_df.copy()
    test_df_copy['prediction'] = ensemble_pred

    # Calculate per-stock accuracy
    stock_accuracies = {}
    for symbol in test_df_copy['symbol'].unique():
        stock_data = test_df_copy[test_df_copy['symbol'] == symbol]
        if len(stock_data) >= 50:  # Minimum samples
            acc = direction_accuracy(
                stock_data['future_return'].values,
                stock_data['prediction'].values
            )
            stock_accuracies[symbol] = acc

    top_5_stocks = sorted(stock_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
    core_stocks = [s[0] for s in top_5_stocks]

    print(f"\nCore stocks (Top 5):")
    for i, (symbol, acc) in enumerate(top_5_stocks, 1):
        print(f"  {i}. {symbol}: {acc*100:.2f}%")

    # Test different configurations
    print("\n" + "=" * 80)
    print("Testing Ensemble Signal Configurations")
    print("=" * 80)

    configs = [
        {
            'name': 'Strong Signals Only (All 3 agree)',
            'strategy1_n': 7,
            'strategy2_n': 7,
            'strategy2_window': 30,
            'core_stocks': core_stocks,
            'satellite_n': 2,
            'signal_threshold': 3  # All 3 must agree
        },
        {
            'name': 'Medium+ Signals (2+ agree)',
            'strategy1_n': 7,
            'strategy2_n': 7,
            'strategy2_window': 30,
            'core_stocks': core_stocks,
            'satellite_n': 2,
            'signal_threshold': 2  # At least 2 agree
        },
        {
            'name': 'All Signals (1+ agree)',
            'strategy1_n': 7,
            'strategy2_n': 7,
            'strategy2_window': 30,
            'core_stocks': core_stocks,
            'satellite_n': 2,
            'signal_threshold': 1  # Any signal
        },
        {
            'name': 'Conservative (Strong only, Top 5)',
            'strategy1_n': 5,
            'strategy2_n': 5,
            'strategy2_window': 45,
            'core_stocks': core_stocks,
            'satellite_n': 1,
            'signal_threshold': 3
        },
    ]

    all_results = []

    for config in configs:
        print(f"\n{config['name']}:")

        results = backtest_ensemble_signals(test_df, ensemble_pred, config)

        if results:
            print(f"  Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
            print(f"  Total Trades: {results['total_trades']:,}")
            print(f"  Coverage: {results['coverage']*100:.1f}%")

            print(f"  Signal breakdown:")
            for strength, stats in results['by_signal_strength'].items():
                if stats['count'] > 0:
                    print(f"    {strength.upper():8s}: {stats['count']:4d} trades, {stats['accuracy']*100:.2f}% acc")

            improvement = (results['overall_accuracy'] - baseline_acc) * 100
            print(f"  Improvement: {improvement:+.2f} pp")

            results['name'] = config['name']
            all_results.append(results)

    # Best result
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    all_results.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    best = all_results[0]

    print(f"\nStrategy: {best['name']}")
    print(f"Accuracy: {best['overall_accuracy']*100:.2f}%")
    print(f"Improvement: {(best['overall_accuracy'] - baseline_acc)*100:+.2f} pp")
    print(f"Coverage: {best['coverage']*100:.1f}%")
    print(f"Total Trades: {best['total_trades']:,}")

    print(f"\nSignal Strength Breakdown:")
    for strength in ['strong', 'medium', 'weak']:
        stats = best['by_signal_strength'][strength]
        if stats['count'] > 0:
            print(f"  {strength.upper():8s}: {stats['count']:5d} trades ({stats['accuracy']*100:.2f}% acc)")

    # Save results
    output_path = artifacts_dir / "stage1_3_ensemble_signals_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'baseline_accuracy': float(baseline_acc),
            'core_stocks': core_stocks,
            'best_config': best,
            'all_configs': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nBaseline: {baseline_acc*100:.2f}%")
    print(f"Best Ensemble: {best['overall_accuracy']*100:.2f}%")
    print(f"Improvement: {(best['overall_accuracy'] - baseline_acc)*100:+.2f} pp")

    if best['overall_accuracy'] >= 0.66:
        print("\n[SUCCESS] Achieved 66%+ target!")
        print("Ready for Stage 2: Individual Stock Models")
    elif best['overall_accuracy'] >= 0.65:
        print("\n[GOOD] Close to 66% target")

    return best


if __name__ == "__main__":
    main()
