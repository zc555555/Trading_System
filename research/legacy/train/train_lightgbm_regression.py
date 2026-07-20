"""
Train LightGBM Regression model to predict tomorrow's exact price change.
Part of the ensemble model system for maximum accuracy.
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def train_lightgbm_regression():
    """Train LightGBM regression model to predict exact price change."""

    if not LIGHTGBM_AVAILABLE:
        print("=" * 80)
        print("ERROR: LightGBM not installed")
        print("=" * 80)
        print("\nPlease install LightGBM:")
        print("  pip install lightgbm")
        return

    print("=" * 80)
    print("TRAINING LIGHTGBM REGRESSION MODEL FOR PRICE PREDICTION")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load dataset
    parquet_path = Path(__file__).parent.parent / "data" / "stocks.parquet"
    if not parquet_path.exists():
        print(f"ERROR: Dataset not found at {parquet_path}")
        print("Please run: python run_build_fixed.py")
        return

    print(f"\nLoading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"\nFeatures: {len(feature_cols)}")

    # Split train/test (80/20 time-based)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    # Prepare data for regression (predict future_return directly)
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train LightGBM Regressor
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM REGRESSOR")
    print("=" * 80)

    # Get params from config
    lgb_params = config['model']['lightgbm'].copy()

    print("\nLightGBM Parameters:")
    for key, value in lgb_params.items():
        print(f"  {key}: {value}")

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_data)

    # Set objective
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'rmse'
    lgb_params['verbose'] = -1

    print("\nTraining...")
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        num_boost_round=lgb_params['n_estimators'],
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress verbose output
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nTrain Set:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  MAE: {train_mae:.6f} ({train_mae*100:.2f}%)")
    print(f"  R2:  {train_r2:.4f}")

    print(f"\nTest Set:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f} ({test_mae*100:.2f}%)")
    print(f"  R2:  {test_r2:.4f}")

    # Directional accuracy (did we predict the right direction?)
    train_direction_correct = ((y_train > 0) == (y_train_pred > 0)).mean()
    test_direction_correct = ((y_test > 0) == (y_test_pred > 0)).mean()

    print(f"\nDirectional Accuracy:")
    print(f"  Train: {train_direction_correct*100:.2f}%")
    print(f"  Test:  {test_direction_correct*100:.2f}%")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n" + "=" * 80)
    print("TOP 15 IMPORTANT FEATURES")
    print("=" * 80)
    for i, (feat, imp) in enumerate(feature_importance[:15], 1):
        print(f"{i:2d}. {feat:20s}: {imp:.0f}")

    # Save model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "lightgbm_regression_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        "model_type": "LightGBM Regressor",
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "test_direction_accuracy": float(test_direction_correct),
        "train_mse": float(train_mse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "train_direction_accuracy": float(train_direction_correct),
        "feature_importance": [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in feature_importance[:20]
        ]
    }

    metrics_path = artifacts_dir / "lightgbm_regression_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("LIGHTGBM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTest MAE: {test_mae*100:.2f}% (average prediction error)")
    print(f"Test R2:  {test_r2:.4f} (higher is better, max 1.0)")
    print(f"Direction Accuracy: {test_direction_correct*100:.2f}%")
    print("\nThis model will be combined with XGBoost and CatBoost for ensemble predictions.")


if __name__ == "__main__":
    train_lightgbm_regression()
