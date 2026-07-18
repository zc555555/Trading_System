"""
Train CatBoost Regression Model with SELECTED FEATURES (60 features).

Expected improvement: 58.38% -> 59-60% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_catboost_selected():
    """Train CatBoost with selected 60 features."""

    print("=" * 80)
    print("TRAINING CATBOOST REGRESSION MODEL (SELECTED 60 FEATURES)")
    print("=" * 80)

    # Load selected features dataset
    parquet_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    if not parquet_path.exists():
        print(f"ERROR: Selected features dataset not found at {parquet_path}")
        print("Please run: python train/select_top_features.py")
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

    # Split data (80/20 time-based)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST REGRESSOR")
    print("=" * 80)

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        bagging_temperature=0.7,
        random_strength=0.1,
        border_count=128,
        random_state=42,
        verbose=False,
        thread_count=-1
    )

    print("\nCatBoost Parameters:")
    print(f"  iterations: {model.get_param('iterations')}")
    print(f"  depth: {model.get_param('depth')}")
    print(f"  learning_rate: {model.get_param('learning_rate')}")
    print(f"  l2_leaf_reg: {model.get_param('l2_leaf_reg')}")

    print("\nTraining...")
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    # Train predictions
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    train_direction = ((y_train > 0) == (train_pred > 0)).mean()

    # Test predictions
    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_direction = ((y_test > 0) == (test_pred > 0)).mean()

    print(f"\nTrain Set:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  MAE: {train_mae:.6f} ({train_mae*100:.2f}%)")
    print(f"  R2:  {train_r2:.4f}")

    print(f"\nTest Set:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f} ({test_mae*100:.2f}%)")
    print(f"  R2:  {test_r2:.4f}")

    print(f"\nDirectional Accuracy:")
    print(f"  Train: {train_direction*100:.2f}%")
    print(f"  Test:  {test_direction*100:.2f}%")

    # Feature importance
    print("\n" + "=" * 80)
    print("TOP 15 IMPORTANT FEATURES")
    print("=" * 80)

    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(15).iterrows():
        print(f"{row.name + 1:2d}. {row['feature']:30s} : {row['importance']:.4f}")

    # Save model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / "catboost_selected_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        "features_count": len(feature_cols),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "test_direction_accuracy": float(test_direction),
        "train_mse": float(train_mse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "train_direction_accuracy": float(train_direction),
        "feature_importance": importance_df.head(20).to_dict('records')
    }

    metrics_path = artifacts_dir / "catboost_selected_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("CATBOOST TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\nTest MAE: {test_mae*100:.2f}% (average prediction error)")
    print(f"Test R2:  {test_r2:.4f} (higher is better, max 1.0)")
    print(f"Direction Accuracy: {test_direction*100:.2f}%")

    print("\nNext step: Create ensemble with selected features")


if __name__ == "__main__":
    train_catboost_selected()
