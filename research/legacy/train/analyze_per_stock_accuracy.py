"""
Analyze direction accuracy for each individual stock.

This helps identify:
1. Which stocks are easier to predict
2. Which stocks should we focus on trading
3. Whether to build stock-specific models
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def analyze_per_stock():
    """Analyze performance for each stock individually."""

    print("=" * 80)
    print("PER-STOCK ACCURACY ANALYSIS")
    print("=" * 80)

    # Load optimized ensemble model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / "ensemble_optimized_weights_model.pkl"

    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)

    models = ensemble['models']
    weights = ensemble['weights']
    feature_cols = ensemble['feature_cols']

    print(f"\nLoaded ensemble model:")
    print(f"  XGBoost:  {weights['xgboost']*100:.1f}%")
    print(f"  LightGBM: {weights['lightgbm']*100:.1f}%")
    print(f"  CatBoost: {weights['catboost']*100:.1f}%")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTest set: {len(test_df):,} samples")
    print(f"Date range: {test_df['date'].min()} to {test_df['date'].max()}")

    # Get predictions for all test samples
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Get predictions from each model
    xgb_pred = models['xgboost'].predict(X_test)
    lgb_pred = models['lightgbm'].predict(X_test)
    cat_pred = models['catboost'].predict(X_test)

    # Ensemble prediction
    ensemble_pred = (weights['xgboost'] * xgb_pred +
                     weights['lightgbm'] * lgb_pred +
                     weights['catboost'] * cat_pred)

    # Add predictions to dataframe
    test_df['ensemble_pred'] = ensemble_pred
    test_df['xgb_pred'] = xgb_pred
    test_df['lgb_pred'] = lgb_pred
    test_df['cat_pred'] = cat_pred

    # Analyze per stock
    print("\n" + "=" * 80)
    print("PER-STOCK PERFORMANCE")
    print("=" * 80)

    results = []

    for symbol in sorted(test_df['symbol'].unique()):
        stock_df = test_df[test_df['symbol'] == symbol].copy()

        y_true = stock_df['future_return'].values
        y_pred = stock_df['ensemble_pred'].values

        # Metrics
        n_samples = len(stock_df)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        direction_acc = direction_accuracy(y_true, y_pred)

        # Prediction stats
        avg_pred = y_pred.mean()
        std_pred = y_pred.std()
        avg_actual = y_true.mean()

        results.append({
            'symbol': symbol,
            'samples': n_samples,
            'direction_accuracy': direction_acc,
            'mae': mae,
            'r2': r2,
            'avg_pred': avg_pred,
            'std_pred': std_pred,
            'avg_actual': avg_actual
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('direction_accuracy', ascending=False)

    # Display results
    print(f"\n{'Symbol':<8} {'Samples':>8} {'Dir.Acc':>9} {'MAE':>8} {'R2':>8} {'Avg.Pred':>10} {'Avg.Actual':>10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        symbol = row['symbol']
        samples = row['samples']
        acc = row['direction_accuracy']
        mae = row['mae']
        r2 = row['r2']
        avg_pred = row['avg_pred']
        avg_actual = row['avg_actual']

        # Color coding for accuracy
        if acc >= 0.62:
            marker = "+++"
        elif acc >= 0.60:
            marker = "++"
        elif acc >= 0.58:
            marker = "+"
        elif acc >= 0.56:
            marker = "="
        else:
            marker = "-"

        print(f"{symbol:<8} {samples:>8} {acc*100:>8.2f}% {mae*100:>7.2f}% {r2:>8.4f} {avg_pred*100:>9.2f}% {avg_actual*100:>9.2f}%  {marker}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nOverall (24 stocks):")
    print(f"  Mean accuracy:   {results_df['direction_accuracy'].mean()*100:.2f}%")
    print(f"  Median accuracy: {results_df['direction_accuracy'].median()*100:.2f}%")
    print(f"  Std deviation:   {results_df['direction_accuracy'].std()*100:.2f}%")
    print(f"  Min accuracy:    {results_df['direction_accuracy'].min()*100:.2f}%")
    print(f"  Max accuracy:    {results_df['direction_accuracy'].max()*100:.2f}%")

    # Group by accuracy level
    print(f"\nAccuracy Distribution:")
    very_high = results_df[results_df['direction_accuracy'] >= 0.62]
    high = results_df[(results_df['direction_accuracy'] >= 0.60) & (results_df['direction_accuracy'] < 0.62)]
    good = results_df[(results_df['direction_accuracy'] >= 0.58) & (results_df['direction_accuracy'] < 0.60)]
    medium = results_df[(results_df['direction_accuracy'] >= 0.56) & (results_df['direction_accuracy'] < 0.58)]
    low = results_df[results_df['direction_accuracy'] < 0.56]

    print(f"  >=62% (Excellent):  {len(very_high)} stocks - {list(very_high['symbol'])}")
    print(f"  60-62% (Very Good): {len(high)} stocks - {list(high['symbol'])}")
    print(f"  58-60% (Good):      {len(good)} stocks - {list(good['symbol'])}")
    print(f"  56-58% (Fair):      {len(medium)} stocks - {list(medium['symbol'])}")
    print(f"  <56% (Poor):        {len(low)} stocks - {list(low['symbol'])}")

    # Top and bottom performers
    print("\n" + "=" * 80)
    print("TOP 5 PERFORMERS (Easiest to Predict)")
    print("=" * 80)

    for i, row in enumerate(results_df.head(5).itertuples(), 1):
        print(f"\n{i}. {row.symbol} - {row.direction_accuracy*100:.2f}% accuracy")
        print(f"   Samples: {row.samples}, MAE: {row.mae*100:.2f}%, R2: {row.r2:.4f}")

    print("\n" + "=" * 80)
    print("BOTTOM 5 PERFORMERS (Hardest to Predict)")
    print("=" * 80)

    for i, row in enumerate(results_df.tail(5).itertuples(), 1):
        print(f"\n{i}. {row.symbol} - {row.direction_accuracy*100:.2f}% accuracy")
        print(f"   Samples: {row.samples}, MAE: {row.mae*100:.2f}%, R2: {row.r2:.4f}")

    # Strategy recommendations
    print("\n" + "=" * 80)
    print("TRADING STRATEGY RECOMMENDATIONS")
    print("=" * 80)

    top_stocks = results_df.head(10)
    avg_top_10 = top_stocks['direction_accuracy'].mean()

    print(f"\nStrategy 1: Trade Top 10 Stocks")
    print(f"  Stocks: {list(top_stocks['symbol'])}")
    print(f"  Expected accuracy: {avg_top_10*100:.2f}%")
    print(f"  Improvement: {(avg_top_10 - 0.5902)*100:+.2f} percentage points")

    top_5_stocks = results_df.head(5)
    avg_top_5 = top_5_stocks['direction_accuracy'].mean()

    print(f"\nStrategy 2: Trade Top 5 Stocks Only")
    print(f"  Stocks: {list(top_5_stocks['symbol'])}")
    print(f"  Expected accuracy: {avg_top_5*100:.2f}%")
    print(f"  Improvement: {(avg_top_5 - 0.5902)*100:+.2f} percentage points")

    # If any stock is >= 62%
    excellent_stocks = results_df[results_df['direction_accuracy'] >= 0.62]
    if len(excellent_stocks) > 0:
        avg_excellent = excellent_stocks['direction_accuracy'].mean()
        print(f"\nStrategy 3: Trade Only Excellent Stocks (>=62%)")
        print(f"  Stocks: {list(excellent_stocks['symbol'])}")
        print(f"  Expected accuracy: {avg_excellent*100:.2f}%")
        print(f"  Improvement: {(avg_excellent - 0.5902)*100:+.2f} percentage points")

    # Save results
    output_path = artifacts_dir / "per_stock_accuracy.json"
    output_data = {
        'overall_accuracy': 0.5902,
        'per_stock': results_df.to_dict('records'),
        'summary': {
            'mean': float(results_df['direction_accuracy'].mean()),
            'median': float(results_df['direction_accuracy'].median()),
            'std': float(results_df['direction_accuracy'].std()),
            'min': float(results_df['direction_accuracy'].min()),
            'max': float(results_df['direction_accuracy'].max())
        },
        'top_10_stocks': list(top_stocks['symbol']),
        'top_10_avg_accuracy': float(avg_top_10)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    analyze_per_stock()
