"""Simple test of dynamic stock selection strategy"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np

def direction_accuracy(y_true, y_pred):
    return ((y_true > 0) == (y_pred > 0)).mean()

# Load model
artifacts_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/artifacts")
model_path = artifacts_dir / "ensemble_time_windows.pkl"

print(f"Loading model: {model_path}")
with open(model_path, 'rb') as f:
    ensemble_dict = pickle.load(f)

models = ensemble_dict['models']
weights = ensemble_dict['weights']
feature_cols = ensemble_dict['feature_cols']

print(f"Models: {list(models.keys())}")
print(f"Features: {len(feature_cols)}")

# Load data
data_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/data")
df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

print(f"\nDataset: {df.shape}")
print(f"Stocks: {df['symbol'].nunique()}")

# Split data
split_date = df['date'].quantile(0.8)
test_df = df[df['date'] > split_date].copy()

print(f"Test set: {len(test_df):,} samples")

# Prepare features
X_test = test_df[feature_cols].values
y_test = test_df['future_return'].values
X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

# Generate predictions
print("\nGenerating predictions...")
predictions = {}
for model_name in sorted(models.keys()):
    model = models[model_name]
    pred = model.predict(X_test)
    predictions[model_name] = pred

pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
ensemble_pred = pred_matrix @ weights_array

# Baseline
baseline_acc = direction_accuracy(y_test, ensemble_pred)
print(f"\nBaseline accuracy (all stocks): {baseline_acc*100:.2f}%")

# Test: Select top 7 stocks by confidence each day
test_df_copy = test_df.copy()
test_df_copy['prediction'] = ensemble_pred
test_df_copy['confidence'] = np.abs(ensemble_pred)

selected_trades = []
for date in test_df_copy['date'].unique():
    day_data = test_df_copy[test_df_copy['date'] == date]
    top_7 = day_data.nlargest(7, 'confidence')
    selected_trades.append(top_7)

all_trades = pd.concat(selected_trades, ignore_index=True)

# Calculate metrics
acc = direction_accuracy(
    all_trades['future_return'].values,
    all_trades['prediction'].values
)

coverage = len(all_trades) / len(test_df_copy)
unique_stocks = all_trades['symbol'].nunique()
trades_per_day = len(all_trades) / test_df_copy['date'].nunique()

print(f"\nDynamic Top 7 Strategy:")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  Improvement: {(acc - baseline_acc)*100:+.2f} pp")
print(f"  Coverage: {coverage*100:.1f}%")
print(f"  Avg trades/day: {trades_per_day:.1f}")
print(f"  Unique stocks: {unique_stocks}")

# Per stock performance
print(f"\nTop stocks by performance:")
per_stock = {}
for symbol in all_trades['symbol'].unique():
    stock_data = all_trades[all_trades['symbol'] == symbol]
    stock_acc = direction_accuracy(
        stock_data['future_return'].values,
        stock_data['prediction'].values
    )
    per_stock[symbol] = {
        'accuracy': stock_acc,
        'trades': len(stock_data)
    }

sorted_stocks = sorted(per_stock.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (symbol, stats) in enumerate(sorted_stocks[:10], 1):
    print(f"  {i:2d}. {symbol:<6} {stats['accuracy']*100:>6.2f}%  ({stats['trades']:>3} trades)")

# Save results
results = {
    'baseline_accuracy': float(baseline_acc),
    'strategy_accuracy': float(acc),
    'improvement': float((acc - baseline_acc) * 100),
    'coverage': float(coverage),
    'unique_stocks': int(unique_stocks),
    'trades_per_day': float(trades_per_day),
    'per_stock_performance': {k: {'accuracy': float(v['accuracy']), 'trades': int(v['trades'])}
                              for k, v in per_stock.items()}
}

output_path = artifacts_dir / "dynamic_strategy_simple_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")
