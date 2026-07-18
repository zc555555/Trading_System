"""
Hyperparameter Optimization for XGBoost, LightGBM, and CatBoost.

Uses RandomizedSearchCV with custom direction accuracy scorer.
Expected improvement: 58.59% -> 60-61% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import time

def direction_accuracy_score(y_true, y_pred):
    """Custom scorer for direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()

def load_data():
    """Load selected features dataset."""
    parquet_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(parquet_path)

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    return X_train, y_train, X_test, y_test, feature_cols


def optimize_xgboost(X_train, y_train, X_test, y_test):
    """Optimize XGBoost hyperparameters."""

    print("\n" + "=" * 80)
    print("OPTIMIZING XGBOOST")
    print("=" * 80)

    # Parameter search space
    param_dist = {
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'n_estimators': [300, 500, 700],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }

    # Base model
    base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

    # Custom scorer
    direction_scorer = make_scorer(direction_accuracy_score, greater_is_better=True)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    print("\nSearching hyperparameters...")
    print(f"  Search space size: {np.prod([len(v) for v in param_dist.values()]):,} combinations")
    print(f"  Trying: 20 random combinations")
    print(f"  CV splits: 3")

    start_time = time.time()

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring=direction_scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time

    print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV direction accuracy: {search.best_score_*100:.2f}%")

    # Evaluate on test set
    best_model = search.best_estimator_
    test_pred = best_model.predict(X_test)
    test_direction = direction_accuracy_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nTest set performance:")
    print(f"  Direction Accuracy: {test_direction*100:.2f}%")
    print(f"  MAE: {test_mae*100:.2f}%")
    print(f"  R2: {test_r2:.4f}")

    return best_model, search.best_params_, test_direction


def optimize_lightgbm(X_train, y_train, X_test, y_test):
    """Optimize LightGBM hyperparameters."""

    print("\n" + "=" * 80)
    print("OPTIMIZING LIGHTGBM")
    print("=" * 80)

    # Parameter search space
    param_dist = {
        'num_leaves': [20, 31, 40, 50],
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'n_estimators': [300, 500, 700],
        'min_child_samples': [10, 20, 30],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
        'bagging_freq': [3, 5, 7],
        'lambda_l1': [0, 0.1, 0.5, 1.0],
        'lambda_l2': [0, 0.1, 0.5, 1.0]
    }

    # Base model
    base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)

    # Custom scorer
    direction_scorer = make_scorer(direction_accuracy_score, greater_is_better=True)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    print("\nSearching hyperparameters...")
    print(f"  Search space size: {np.prod([len(v) for v in param_dist.values()]):,} combinations")
    print(f"  Trying: 20 random combinations")
    print(f"  CV splits: 3")

    start_time = time.time()

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring=direction_scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time

    print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV direction accuracy: {search.best_score_*100:.2f}%")

    # Evaluate on test set
    best_model = search.best_estimator_
    test_pred = best_model.predict(X_test)
    test_direction = direction_accuracy_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nTest set performance:")
    print(f"  Direction Accuracy: {test_direction*100:.2f}%")
    print(f"  MAE: {test_mae*100:.2f}%")
    print(f"  R2: {test_r2:.4f}")

    return best_model, search.best_params_, test_direction


def optimize_catboost(X_train, y_train, X_test, y_test):
    """Optimize CatBoost hyperparameters."""

    print("\n" + "=" * 80)
    print("OPTIMIZING CATBOOST")
    print("=" * 80)

    # Parameter search space
    param_dist = {
        'depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'iterations': [300, 500, 700],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [64, 128, 255],
        'bagging_temperature': [0.5, 0.7, 1.0],
        'random_strength': [0.1, 0.5, 1.0]
    }

    # Base model
    base_model = CatBoostRegressor(random_state=42, verbose=False, thread_count=-1)

    # Custom scorer
    direction_scorer = make_scorer(direction_accuracy_score, greater_is_better=True)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    print("\nSearching hyperparameters...")
    print(f"  Search space size: {np.prod([len(v) for v in param_dist.values()]):,} combinations")
    print(f"  Trying: 20 random combinations")
    print(f"  CV splits: 3")

    start_time = time.time()

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring=direction_scorer,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    elapsed = time.time() - start_time

    print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV direction accuracy: {search.best_score_*100:.2f}%")

    # Evaluate on test set
    best_model = search.best_estimator_
    test_pred = best_model.predict(X_test)
    test_direction = direction_accuracy_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nTest set performance:")
    print(f"  Direction Accuracy: {test_direction*100:.2f}%")
    print(f"  MAE: {test_mae*100:.2f}%")
    print(f"  R2: {test_r2:.4f}")

    return best_model, search.best_params_, test_direction


def main():
    """Main hyperparameter optimization workflow."""

    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\nOptimizing XGBoost, LightGBM, and CatBoost for direction accuracy")
    print("Expected improvement: 58.59% -> 60-61%")
    print("\nThis will take approximately 2-3 hours...")

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")

    # Optimize each model
    results = {}

    # XGBoost
    xgb_model, xgb_params, xgb_acc = optimize_xgboost(X_train, y_train, X_test, y_test)
    results['xgboost'] = {
        'model': xgb_model,
        'params': xgb_params,
        'test_direction_accuracy': float(xgb_acc)
    }

    # LightGBM
    lgb_model, lgb_params, lgb_acc = optimize_lightgbm(X_train, y_train, X_test, y_test)
    results['lightgbm'] = {
        'model': lgb_model,
        'params': lgb_params,
        'test_direction_accuracy': float(lgb_acc)
    }

    # CatBoost
    cb_model, cb_params, cb_acc = optimize_catboost(X_train, y_train, X_test, y_test)
    results['catboost'] = {
        'model': cb_model,
        'params': cb_params,
        'test_direction_accuracy': float(cb_acc)
    }

    # Save optimized models
    print("\n" + "=" * 80)
    print("SAVING OPTIMIZED MODELS")
    print("=" * 80)

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Save models
    for model_name, data in results.items():
        model_path = artifacts_dir / f"{model_name}_optimized_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(data['model'], f)
        print(f"  [OK] {model_name} saved to {model_path.name}")

    # Save parameters
    params_data = {
        'xgboost': results['xgboost']['params'],
        'lightgbm': results['lightgbm']['params'],
        'catboost': results['catboost']['params']
    }

    params_path = artifacts_dir / "optimized_hyperparameters.json"
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params_data, f, indent=2)
    print(f"  [OK] Parameters saved to {params_path.name}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

    print("\nTest Direction Accuracy:")
    print(f"  XGBoost:  {results['xgboost']['test_direction_accuracy']*100:.2f}%")
    print(f"  LightGBM: {results['lightgbm']['test_direction_accuracy']*100:.2f}%")
    print(f"  CatBoost: {results['catboost']['test_direction_accuracy']*100:.2f}%")

    avg_acc = np.mean([
        results['xgboost']['test_direction_accuracy'],
        results['lightgbm']['test_direction_accuracy'],
        results['catboost']['test_direction_accuracy']
    ])

    print(f"\nAverage: {avg_acc*100:.2f}%")
    print(f"Improvement from 58.59%: {(avg_acc - 0.5859)*100:+.2f} percentage points")

    print("\nNext step: Create ensemble with optimized models")
    print("  python train/train_ensemble_optimized.py")


if __name__ == "__main__":
    main()
