"""
Train models with Sample Weighting (Recent data gets higher weight).

Expected improvement: +0.5-1% direction accuracy
Strategy: Exponential decay - recent data is more important
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def get_sample_weights(dates, half_life_days=180):
    """
    Calculate exponential decay weights for samples.

    Recent data gets higher weight.
    Half-life: weight decreases by 50% every 'half_life_days' days.

    Args:
        dates: pandas Series of dates
        half_life_days: Number of days for weight to decay to 50%

    Returns:
        numpy array of weights
    """
    # Convert dates to days from start
    days_from_start = (dates - dates.min()).dt.days.values
    max_days = days_from_start.max()

    # Exponential decay: recent data = higher weight
    # Formula: weight = exp((days - max_days) / half_life)
    decay_rate = half_life_days / np.log(2)
    weights = np.exp((days_from_start - max_days) / decay_rate)

    # Normalize to mean=1 (so total weight = n_samples)
    weights = weights / weights.mean()

    return weights


def train_weighted():
    """Train models with sample weighting."""

    print("=" * 80)
    print("SAMPLE-WEIGHTED TRAINING")
    print("=" * 80)
    print("\nStrategy: Recent data gets exponentially higher weight")
    print("Half-life: 180 days (weight halves every 6 months)")
    print("Expected improvement: +0.5-1.0% direction accuracy")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    print(f"\nDataset: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Features: {len(feature_cols)}")

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values
    train_dates = train_df['date']

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Calculate sample weights
    print("\n" + "=" * 80)
    print("CALCULATING SAMPLE WEIGHTS")
    print("=" * 80)

    sample_weights = get_sample_weights(train_dates, half_life_days=180)

    print(f"\nWeight statistics:")
    print(f"  Mean:   {sample_weights.mean():.4f}")
    print(f"  Median: {np.median(sample_weights):.4f}")
    print(f"  Min:    {sample_weights.min():.4f}")
    print(f"  Max:    {sample_weights.max():.4f}")
    print(f"  Std:    {sample_weights.std():.4f}")

    # Show weight distribution over time
    weight_by_year = train_df.copy()
    weight_by_year['weight'] = sample_weights
    weight_by_year['year'] = weight_by_year['date'].dt.year

    print(f"\nAverage weight by year:")
    for year in sorted(weight_by_year['year'].unique()):
        avg_weight = weight_by_year[weight_by_year['year'] == year]['weight'].mean()
        print(f"  {year}: {avg_weight:.4f}")

    # Train XGBoost with weights
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST (WITH WEIGHTS)")
    print("=" * 80)

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    print("\nTraining XGBoost with sample weights...")
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_acc = direction_accuracy(y_test, xgb_test_pred)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)

    print(f"  Test Direction Accuracy: {xgb_test_acc*100:.2f}%")
    print(f"  Test MAE: {xgb_test_mae*100:.2f}%")

    # Train LightGBM with weights
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM (WITH WEIGHTS)")
    print("=" * 80)

    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }

    print("\nTraining LightGBM with sample weights...")

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500
    )

    lgb_test_pred = lgb_model.predict(X_test)
    lgb_test_acc = direction_accuracy(y_test, lgb_test_pred)
    lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)

    print(f"  Test Direction Accuracy: {lgb_test_acc*100:.2f}%")
    print(f"  Test MAE: {lgb_test_mae*100:.2f}%")

    # Train CatBoost with weights
    print("\n" + "=" * 80)
    print("TRAINING CATBOOST (WITH WEIGHTS)")
    print("=" * 80)

    cb_model = CatBoostRegressor(
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

    print("\nTraining CatBoost with sample weights...")
    cb_model.fit(X_train, y_train, sample_weight=sample_weights)

    cb_test_pred = cb_model.predict(X_test)
    cb_test_acc = direction_accuracy(y_test, cb_test_pred)
    cb_test_mae = mean_absolute_error(y_test, cb_test_pred)

    print(f"  Test Direction Accuracy: {cb_test_acc*100:.2f}%")
    print(f"  Test MAE: {cb_test_mae*100:.2f}%")

    # Ensemble (equal weights for now)
    print("\n" + "=" * 80)
    print("WEIGHTED ENSEMBLE EVALUATION")
    print("=" * 80)

    ensemble_pred = (xgb_test_pred + lgb_test_pred + cb_test_pred) / 3
    ensemble_acc = direction_accuracy(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

    print(f"\nEnsemble (Equal Weights):")
    print(f"  Direction Accuracy: {ensemble_acc*100:.2f}%")
    print(f"  MAE: {ensemble_mae*100:.2f}%")

    # Compare with non-weighted baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    baseline_acc = 0.5859  # From train_ensemble_selected.py

    print(f"\nBaseline (No Weights):    {baseline_acc*100:.2f}%")
    print(f"Sample-Weighted Ensemble: {ensemble_acc*100:.2f}%")
    print(f"Improvement: {(ensemble_acc - baseline_acc)*100:+.2f} percentage points")

    # Save models
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    models = {
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cb_model,
        'weights': {'xgboost': 1/3, 'lightgbm': 1/3, 'catboost': 1/3},
        'feature_cols': feature_cols,
        'sample_weighting': {'half_life_days': 180}
    }

    model_path = artifacts_dir / "ensemble_sample_weighted_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)

    print(f"\nModels saved to: {model_path}")

    # Save metrics
    metrics = {
        'sample_weighting': True,
        'half_life_days': 180,
        'models': {
            'xgboost': {
                'test_direction_accuracy': float(xgb_test_acc),
                'test_mae': float(xgb_test_mae)
            },
            'lightgbm': {
                'test_direction_accuracy': float(lgb_test_acc),
                'test_mae': float(lgb_test_mae)
            },
            'catboost': {
                'test_direction_accuracy': float(cb_test_acc),
                'test_mae': float(cb_test_mae)
            }
        },
        'ensemble': {
            'test_direction_accuracy': float(ensemble_acc),
            'test_mae': float(ensemble_mae)
        },
        'baseline_comparison': {
            'baseline_accuracy': baseline_acc,
            'weighted_accuracy': float(ensemble_acc),
            'improvement': float(ensemble_acc - baseline_acc)
        }
    }

    metrics_path = artifacts_dir / "sample_weighted_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("SAMPLE-WEIGHTED TRAINING COMPLETE!")
    print("=" * 80)

    if ensemble_acc >= 0.60:
        print(f"\n[SUCCESS] Achieved {ensemble_acc*100:.2f}% (>=60%) direction accuracy!")
    else:
        print(f"\n[RESULT] Achieved {ensemble_acc*100:.2f}% direction accuracy.")


if __name__ == "__main__":
    train_weighted()
