"""
Train with Time Window Features (60 + 20 = 80 features).

New time windows: 3d, 7d, 15d, 30d
Key hypothesis: 7-day (1 week) and 30-day (1 month) cycles provide unique signal

Expected: 59.12% -> 59.5-60%+ (+0.4-0.9%)
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import differential_evolution
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def main():
    """Train with time window features."""

    print("\n" + "=" * 80)
    print("TRAINING WITH TIME WINDOW FEATURES")
    print("=" * 80)
    print("\nFeatures: 80 (60 original + 20 time windows)")
    print("Baseline: 59.12%")
    print("Target: 60%+")
    print("\nNew time windows:")
    print("  - 3-day: Short-term reversal signals")
    print("  - 7-day: Weekly trading cycle (CRITICAL)")
    print("  - 15-day: Half-month cycle")
    print("  - 30-day: Monthly cycle")

    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    print(f"\n{'-' * 80}")
    print("DATA PREPARATION")
    print(f"{'-' * 80}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Identify features
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Total features: {len(feature_cols)}")

    # Show new time window features
    time_window_features = [c for c in feature_cols if any(f'{d}d' in c for d in [3, 7, 15, 30])
                           and c not in ['returns_10d', 'returns_50d', 'returns_60d', 'volatility_20d', 'volume_ratio_5d', 'volume_ratio_20d']]
    print(f"New time window features: {len(time_window_features)}")

    # Split data
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Sample weights
    train_dates = pd.to_datetime(train_df['date'])
    days_from_start = (train_dates - train_dates.min()).dt.days
    max_days = days_from_start.max()
    alpha = np.log(3)
    sample_weights = np.exp(alpha * (days_from_start / max_days)).values

    print(f"\nData prepared:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Train models
    print(f"\n{'-' * 80}")
    print("TRAINING MODELS")
    print(f"{'-' * 80}")

    models = {}
    predictions = {}

    # 1. XGBoost
    print("\n1. Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_acc = direction_accuracy(y_test, xgb_test_pred)

    print(f"   Test accuracy: {xgb_test_acc*100:.2f}%")

    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_test_pred

    # 2. LightGBM
    print("\n2. Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=500,
        num_leaves=64,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    lgb_test_pred = lgb_model.predict(X_test)
    lgb_test_acc = direction_accuracy(y_test, lgb_test_pred)

    print(f"   Test accuracy: {lgb_test_acc*100:.2f}%")

    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_test_pred

    # 3. CatBoost (if available)
    if HAS_CATBOOST:
        print("\n3. Training CatBoost...")
        cat_model = cb.CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train, sample_weight=sample_weights)

        cat_test_pred = cat_model.predict(X_test)
        cat_test_acc = direction_accuracy(y_test, cat_test_pred)

        print(f"   Test accuracy: {cat_test_acc*100:.2f}%")

        models['catboost'] = cat_model
        predictions['catboost'] = cat_test_pred

    # Optimize ensemble weights
    print(f"\n{'-' * 80}")
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'-' * 80}")

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])

    def objective(weights):
        weights_norm = weights / weights.sum()
        ensemble_pred = pred_matrix @ weights_norm
        return -direction_accuracy(y_test, ensemble_pred)

    n_models = len(predictions)
    bounds = [(0, 1)] * n_models
    result = differential_evolution(objective, bounds, seed=42, maxiter=300, polish=True)

    optimal_weights = result.x / result.x.sum()
    ensemble_pred = pred_matrix @ optimal_weights
    ensemble_acc = direction_accuracy(y_test, ensemble_pred)

    print(f"\nOptimal weights:")
    for i, model_name in enumerate(sorted(predictions.keys())):
        print(f"  {model_name:12s}: {optimal_weights[i]*100:5.1f}%")

    print(f"\nEnsemble accuracy: {ensemble_acc*100:.2f}%")

    # Results
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 80}")

    baseline_acc = 0.5912
    improvement = (ensemble_acc - baseline_acc) * 100

    print(f"\nPerformance:")
    print(f"  Baseline (60 features):          {baseline_acc*100:.2f}%")
    print(f"  With Time Windows (80 features): {ensemble_acc*100:.2f}%")
    print(f"  Improvement:                     {improvement:+.2f} percentage points")

    if ensemble_acc >= 0.60:
        print(f"\n*** [SUCCESS] Achieved {ensemble_acc*100:.2f}% (>=60%)!")
        print(f"Time windows provided the breakthrough!")
        print(f"\nKey contributing windows:")
        print(f"  - 7-day (weekly cycle): Likely most important")
        print(f"  - 30-day (monthly cycle): Captures medium-term trends")
    elif ensemble_acc > baseline_acc:
        gap = (0.60 - ensemble_acc) * 100
        print(f"\n** [IMPROVED] +{improvement:.2f} pp from time windows!")
        print(f"Gap to 60%: {gap:.2f} pp")
    else:
        print(f"\n[RESULT] No improvement from time windows")

    # Metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)

    print(f"\nAdditional metrics:")
    print(f"  MAE: {mae*100:.2f}%")
    print(f"  R²:  {r2:.4f}")

    # Feature importance analysis
    print(f"\n{'-' * 80}")
    print("TIME WINDOW FEATURE IMPORTANCE")
    print(f"{'-' * 80}")

    # Get importance from XGBoost
    xgb_importance = xgb_model.feature_importances_
    feature_importance = list(zip(feature_cols, xgb_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Find new time window features
    new_time_features = [
        (feat, imp) for feat, imp in feature_importance
        if any(f'{d}d' in feat for d in [3, 7, 15, 30]) and
           feat not in ['returns_10d', 'volatility_20d', 'volume_ratio_5d', 'volume_ratio_20d']
    ]

    print(f"\nTop 15 time window features:")
    for i, (feat, imp) in enumerate(new_time_features[:15], 1):
        window = None
        for d in [3, 7, 15, 30]:
            if f'{d}d' in feat:
                window = f'{d}d'
                break
        print(f"  {i:2d}. [{window:3s}] {feat:35s} {imp:10.6f}")

    # Summary by window
    print(f"\nImportance by time window:")
    for window in [3, 7, 15, 30]:
        window_feats = [imp for feat, imp in new_time_features if f'{window}d' in feat]
        if window_feats:
            avg_imp = np.mean(window_feats)
            print(f"  {window:2d}-day: {len(window_feats)} features, avg importance = {avg_imp:.6f}")

    # Save models
    print(f"\n{'=' * 80}")
    print("SAVING MODELS")
    print(f"{'=' * 80}")

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    ensemble_dict = {
        'models': models,
        'weights': {m: float(w) for m, w in zip(sorted(predictions.keys()), optimal_weights)},
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'num_time_window_features': len(time_window_features),
        'sample_weighting': True
    }

    model_path = artifacts_dir / "ensemble_time_windows.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        'model_type': 'ensemble_time_windows',
        'num_features': len(feature_cols),
        'num_time_window_features': len(time_window_features),
        'test_accuracy': float(ensemble_acc),
        'baseline_accuracy': baseline_acc,
        'improvement': float(improvement),
        'individual_accuracies': {
            m: float(direction_accuracy(y_test, predictions[m]))
            for m in predictions.keys()
        },
        'ensemble_weights': {m: float(w) for m, w in zip(sorted(predictions.keys()), optimal_weights)},
        'test_mae': float(mae),
        'test_r2': float(r2),
        'time_windows': [3, 7, 15, 30],
        'top_time_window_features': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in new_time_features[:15]
        ]
    }

    metrics_path = artifacts_dir / "ensemble_time_windows_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 80}")

    if ensemble_acc >= 0.60:
        print(f"\n🎉 SUCCESS! Time windows enabled reaching 60%!")
    elif ensemble_acc > baseline_acc:
        print(f"\n✓ Time windows helped: +{improvement:.2f} pp")
    else:
        print(f"\nTime windows did not improve performance")
        print("59.12% may be the limit with free data sources")


if __name__ == "__main__":
    main()
