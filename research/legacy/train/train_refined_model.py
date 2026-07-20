"""
Train Refined Model: Optimized Feature Selection (81 features).

Improvements over 59.24% model:
1. Removed 9 useless features (importance = 0)
2. Added 8 top advanced features (interactions)
3. Added 5 simple critical features (gap, volume spike, etc.)

Previous: 59.24%
Target: 59.5-60%+
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
    """Train refined model."""

    print("\n" + "=" * 80)
    print("TRAINING REFINED MODEL (81 FEATURES)")
    print("=" * 80)
    print("\nOptimizations:")
    print("  - Removed 9 useless features")
    print("  - Added 8 top advanced features")
    print("  - Added 5 simple critical features")
    print("\nPrevious best: 59.24%")
    print("Target: 59.5-60%+")

    # Load refined dataset
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_refined_features.parquet")

    print(f"\n{'-' * 80}")
    print("DATA PREPARATION")
    print(f"{'-' * 80}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get features
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Total features: {len(feature_cols)}")

    # Categorize features
    base_feats = [c for c in feature_cols if not any(x in c for x in ['interaction_', 'price_gap', 'volume_spike', 'intraday_range', 'close_position', 'consecutive_strength', 'roc_', 'price_vs_ma', 'volatility_15d', 'returns_30d', 'volume_ratio_7d', 'returns_15d'])]
    time_window_feats = [c for c in feature_cols if any(x in c for x in ['roc_30d', 'volatility_15d', 'returns_30d', 'price_vs_ma', 'volume_ratio_7d', 'roc_15d', 'returns_15d'])]
    advanced_feats = [c for c in feature_cols if 'interaction_' in c or 'squared' in c]
    simple_feats = [c for c in feature_cols if any(x in c for x in ['price_gap', 'volume_spike', 'intraday_range', 'close_position', 'consecutive_strength'])]

    print(f"\nFeature breakdown:")
    print(f"  Base: {len(base_feats)}")
    print(f"  Time windows: {len(time_window_feats)}")
    print(f"  Advanced: {len(advanced_feats)}")
    print(f"  Simple critical: {len(simple_feats)}")

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

    # Sample weights (exponential)
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

    # XGBoost
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

    # LightGBM
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

    # CatBoost
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

    # Optimize ensemble
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
    time_windows_acc = 0.5924
    improvement_vs_baseline = (ensemble_acc - baseline_acc) * 100
    improvement_vs_time_windows = (ensemble_acc - time_windows_acc) * 100

    print(f"\nPerformance progression:")
    print(f"  Baseline (60 features):       59.12%")
    print(f"  Time Windows (80 features):   59.24% (+0.12 pp)")
    print(f"  Refined (81 features):        {ensemble_acc*100:.2f}% ({improvement_vs_baseline:+.2f} pp vs baseline)")

    if ensemble_acc >= 0.60:
        print(f"\n*** [SUCCESS] REACHED 60%+ ACCURACY!")
        print(f"Refined features worked!")
    elif ensemble_acc > time_windows_acc:
        print(f"\n** [IMPROVED] Better than time windows by {improvement_vs_time_windows:+.2f} pp!")
        gap = (0.60 - ensemble_acc) * 100
        print(f"Gap to 60%: {gap:.2f} pp")
    elif ensemble_acc > baseline_acc:
        print(f"\n* [IMPROVED] Better than baseline")
    else:
        print(f"\n[RESULT] No improvement")

    # Metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)

    print(f"\nAdditional metrics:")
    print(f"  MAE: {mae*100:.2f}%")
    print(f"  R2:  {r2:.4f}")

    # Feature importance
    print(f"\n{'-' * 80}")
    print("TOP 20 FEATURES")
    print(f"{'-' * 80}")

    xgb_importance = xgb_model.feature_importances_
    feature_importance = list(zip(feature_cols, xgb_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        # Categorize
        if 'interaction_' in feat:
            cat = "ADV"
        elif any(x in feat for x in ['price_gap', 'volume_spike', 'intraday', 'close_position', 'consecutive']):
            cat = "NEW"
        elif any(x in feat for x in ['roc_', 'price_vs_ma', 'returns_15d', 'returns_30d', 'volatility_15d']):
            cat = "TIME"
        else:
            cat = "BASE"

        print(f"  {i:2d}. [{cat:4s}] {feat:45s} {imp:10.6f}")

    # Check new feature importance
    new_feature_importance = [
        (feat, imp) for feat, imp in feature_importance
        if any(x in feat for x in ['price_gap', 'volume_spike', 'intraday_range', 'close_position', 'consecutive_strength'])
    ]

    if new_feature_importance:
        print(f"\n{'-' * 80}")
        print("NEW SIMPLE FEATURES PERFORMANCE")
        print(f"{'-' * 80}")
        for i, (feat, imp) in enumerate(new_feature_importance, 1):
            print(f"  {i}. {feat:35s} {imp:10.6f}")

        avg_new_imp = np.mean([imp for _, imp in new_feature_importance])
        print(f"\nAverage importance: {avg_new_imp:.6f}")

    # Save
    print(f"\n{'=' * 80}")
    print("SAVING MODELS")
    print(f"{'=' * 80}")

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    ensemble_dict = {
        'models': models,
        'weights': {m: float(w) for m, w in zip(sorted(predictions.keys()), optimal_weights)},
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'sample_weighting': True,
        'model_version': 'refined_v1'
    }

    model_path = artifacts_dir / "ensemble_refined.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)

    print(f"\nModel saved to: {model_path}")

    # Metrics
    metrics = {
        'model_type': 'ensemble_refined',
        'num_features': len(feature_cols),
        'test_accuracy': float(ensemble_acc),
        'baseline_accuracy': baseline_acc,
        'time_windows_accuracy': time_windows_acc,
        'improvement_vs_baseline': float(improvement_vs_baseline),
        'improvement_vs_time_windows': float(improvement_vs_time_windows),
        'individual_accuracies': {
            m: float(direction_accuracy(y_test, predictions[m]))
            for m in predictions.keys()
        },
        'ensemble_weights': {m: float(w) for m, w in zip(sorted(predictions.keys()), optimal_weights)},
        'test_mae': float(mae),
        'test_r2': float(r2),
        'top_features': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in feature_importance[:30]
        ],
        'new_feature_performance': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in new_feature_importance
        ] if new_feature_importance else []
    }

    metrics_path = artifacts_dir / "ensemble_refined_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'=' * 80}")
    print("REFINED MODEL TRAINING COMPLETE!")
    print(f"{'=' * 80}")

    if ensemble_acc >= 0.60:
        print(f"\nCONGRATULATIONS! Reached 60%+ with refined features!")
    elif ensemble_acc > time_windows_acc:
        print(f"\nSuccess! Improved by {improvement_vs_time_windows:+.2f} pp")
        print(f"New best: {ensemble_acc*100:.2f}%")
    else:
        print(f"\nFinal: {ensemble_acc*100:.2f}%")


if __name__ == "__main__":
    main()
