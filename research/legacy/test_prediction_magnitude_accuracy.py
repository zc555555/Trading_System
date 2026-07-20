"""
Test how accurate the predicted magnitude (percentage) is, not just direction.

We want to know:
- If model predicts +3%, how close is the actual return?
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def test_magnitude_accuracy():
    """Test prediction magnitude accuracy for top stocks."""

    print("=" * 80)
    print("TESTING PREDICTION MAGNITUDE ACCURACY")
    print("=" * 80)

    # Paths
    artifacts_dir = Path(__file__).parent / "artifacts"
    data_dir = Path(__file__).parent / "data"

    # Load model
    print("\nLoading model...")
    with open(artifacts_dir / "ensemble_time_windows.pkl", 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # Get latest data for stock selection
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()

    # Predict on latest
    X_latest = latest_data[feature_cols].values
    X_latest = pd.DataFrame(X_latest, columns=feature_cols).ffill().fillna(0).values

    predictions = {}
    for model_name in sorted(models.keys()):
        pred = models[model_name].predict(X_latest)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    latest_data['prediction'] = ensemble_pred
    latest_data['confidence'] = np.abs(ensemble_pred)

    # Select top 10 stocks (same logic as signals)
    sorted_data = latest_data.sort_values('confidence', ascending=False)
    top_10 = sorted_data.head(10)

    print("\n" + "=" * 80)
    print("TOP 10 STOCKS FOR TESTING")
    print("=" * 80)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i}. {row['symbol']:6s} - Predicted: {row['prediction']*100:+6.2f}%")

    # For each top stock, test magnitude accuracy on historical data
    print("\n" + "=" * 80)
    print("MAGNITUDE ACCURACY BY STOCK")
    print("=" * 80)

    results = []

    for symbol in top_10['symbol'].values:
        stock_df = df[df['symbol'] == symbol].copy()
        stock_df = stock_df.sort_values('date')

        # Calculate actual returns
        stock_df['actual_return'] = stock_df['close'].pct_change().shift(-1)

        # Make predictions
        X_stock = stock_df[feature_cols].values
        X_stock = pd.DataFrame(X_stock, columns=feature_cols).ffill().fillna(0).values

        pred_stock = np.zeros(len(X_stock))
        for model_name in sorted(models.keys()):
            pred_stock += models[model_name].predict(X_stock) * weights[model_name]

        stock_df['pred'] = pred_stock

        # Remove NaN
        stock_df = stock_df.dropna(subset=['actual_return'])

        # Calculate metrics
        y_true = stock_df['actual_return'].values
        y_pred = stock_df['pred'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Direction accuracy
        direction_acc = ((y_true > 0) == (y_pred > 0)).mean() * 100

        # MAPE (Mean Absolute Percentage Error)
        # Only calculate for returns != 0 to avoid division by zero
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan

        results.append({
            'symbol': symbol,
            'direction_acc': direction_acc,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'n_samples': len(stock_df)
        })

        print(f"\n{symbol}:")
        print(f"  Direction Accuracy: {direction_acc:.2f}%")
        print(f"  MAE (Mean Absolute Error): {mae*100:.2f}%")
        print(f"  RMSE (Root Mean Squared Error): {rmse*100:.2f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE (Mean Absolute % Error): {mape:.2f}%" if not np.isnan(mape) else "  MAPE: N/A")
        print(f"  Samples: {len(stock_df)}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (Top 10 Stocks Average)")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    print(f"\nDirection Accuracy: {results_df['direction_acc'].mean():.2f}%")
    print(f"MAE (Magnitude Error): {results_df['mae'].mean()*100:.2f}%")
    print(f"RMSE: {results_df['rmse'].mean()*100:.2f}%")
    print(f"R² Score: {results_df['r2'].mean():.4f}")
    print(f"MAPE: {results_df['mape'].mean():.2f}%")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    avg_mae = results_df['mae'].mean() * 100
    avg_direction = results_df['direction_acc'].mean()

    print(f"""
方向准确率: {avg_direction:.1f}%
  → 模型在 {avg_direction:.1f}% 的时间能正确预测涨跌方向

幅度误差 (MAE): {avg_mae:.2f}%
  → 平均来说，预测的涨跌幅度偏差约 {avg_mae:.2f}%
  → 例如：预测涨3%，实际可能是涨 {3-avg_mae:.2f}% 到 {3+avg_mae:.2f}%

R² Score: {results_df['r2'].mean():.4f}
  → 模型能解释 {results_df['r2'].mean()*100:.1f}% 的收益率变化
  → {'较低' if results_df['r2'].mean() < 0.1 else '中等' if results_df['r2'].mean() < 0.3 else '较好'}的拟合度
    """)

    if avg_mae > 2.0:
        print(f"""
⚠️  结论：
  - 方向预测 ({avg_direction:.1f}%) 比较可靠
  - 幅度预测误差 ({avg_mae:.2f}%) 较大
  - 建议：按历史准确率分配仓位，而不是预测幅度
        """)
    else:
        print(f"""
✅ 结论：
  - 方向和幅度预测都比较可靠
  - 可以考虑同时使用准确率和预测幅度来分配仓位
        """)

    return results_df


if __name__ == "__main__":
    results = test_magnitude_accuracy()
