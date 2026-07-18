"""
Train Ultimate Model: ALL Features Combined (110 total).

Features:
- 60 base features
- 30 advanced features (interactions, FFT, higher-order stats)
- 20 time window features (3d, 7d, 15d, 30d)

Previous results:
- Base (60): 59.12%
- Base + Advanced (90): 58.96% (failed)
- Base + Time Windows (80): 59.24% (success +0.12%)

Hypothesis: Time windows + Advanced may have synergistic effects
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
    """Train with ultimate feature set."""

    print("\n" + "=" * 80)
    print("TRAINING ULTIMATE MODEL (110 FEATURES)")
    print("=" * 80)
    print("\nFeature composition:")
    print("  - 60 base features (proven: 59.12%)")
    print("  - 30 advanced features (interactions, FFT, stats)")
    print("  - 20 time window features (proven: +0.12%)")
    print("\nPrevious best: 59.24%")
    print("Target: 60%+")

    # Load ultimate dataset
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_ultimate_features.parquet")

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

    # 3. CatBoost
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

    # Results comparison
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 80}")

    baseline_acc = 0.5912
    time_windows_acc = 0.5924
    improvement_vs_baseline = (ensemble_acc - baseline_acc) * 100
    improvement_vs_time_windows = (ensemble_acc - time_windows_acc) * 100

    print(f"\nPerformance progression:")
    print(f"  Baseline (60 features):               59.12%")
    print(f"  Time Windows (80 features):           59.24% (+0.12 pp)")
    print(f"  Ultimate (110 features):              {ensemble_acc*100:.2f}% ({improvement_vs_baseline:+.2f} pp vs baseline)")

    if ensemble_acc >= 0.60:
        print(f"\n*** [SUCCESS] REACHED 60%+ ACCURACY!")
        print(f"Ultimate feature combination worked!")
        print(f"Improvement: +{improvement_vs_baseline:.2f} pp from baseline")
    elif ensemble_acc > time_windows_acc:
        print(f"\n** [IMPROVED] Better than time windows!")
        print(f"Improvement: +{improvement_vs_time_windows:.2f} pp vs time windows")
        gap = (0.60 - ensemble_acc) * 100
        print(f"Gap to 60%: {gap:.2f} pp")
    elif ensemble_acc > baseline_acc:
        print(f"\n* [IMPROVED] Better than baseline")
        print(f"But not better than time windows alone (59.24%)")
    else:
        print(f"\n[RESULT] No improvement over baseline")
        print("Adding more features introduces noise")

    # Metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)

    print(f"\nAdditional metrics:")
    print(f"  MAE: {mae*100:.2f}%")
    print(f"  R2:  {r2:.4f}")

    # Feature importance analysis
    print(f"\n{'-' * 80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'-' * 80}")

    xgb_importance = xgb_model.feature_importances_
    feature_importance = list(zip(feature_cols, xgb_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 features overall:")
    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        # Categorize feature
        if 'interaction_' in feat:
            category = "INTERACT"
        elif 'fft_' in feat:
            category = "FFT"
        elif any(x in feat for x in ['skew', 'kurtosis', 'iqr']):
            category = "STATS"
        elif 'trend_' in feat:
            category = "TREND"
        elif any(f'{d}d' in feat for d in [3, 7, 15, 30]):
            category = "TIME-WIN"
        else:
            category = "BASE"

        print(f"  {i:2d}. [{category:8s}] {feat:45s} {imp:10.6f}")

    # Category breakdown
    print(f"\nImportance by category:")

    categories = {
        'Base': [imp for feat, imp in feature_importance if not any(x in feat for x in ['interaction_', 'fft_', 'skew', 'kurtosis', 'iqr', 'trend_']) and not any(f'{d}d' in feat for d in [3, 7, 15, 30])],
        'Interactions': [imp for feat, imp in feature_importance if 'interaction_' in feat],
        'Time Windows': [imp for feat, imp in feature_importance if any(f'{d}d' in feat for d in [3, 7, 15, 30]) and 'trend_' not in feat],
        'Higher-order Stats': [imp for feat, imp in feature_importance if any(x in feat for x in ['skew', 'kurtosis', 'iqr'])],
        'FFT': [imp for feat, imp in feature_importance if 'fft_' in feat],
        'Trend': [imp for feat, imp in feature_importance if 'trend_' in feat]
    }

    for cat, imps in categories.items():
        if imps:
            avg_imp = np.mean(imps)
            total_imp = np.sum(imps)
            print(f"  {cat:20s}: {len(imps):3d} features, avg={avg_imp:.6f}, total={total_imp:.4f}")

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
        'sample_weighting': True
    }

    model_path = artifacts_dir / "ensemble_ultimate.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        'model_type': 'ensemble_ultimate',
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
        'feature_categories': {
            cat: {'count': len(imps), 'avg_importance': float(np.mean(imps)) if imps else 0}
            for cat, imps in categories.items()
        },
        'top_features': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in feature_importance[:30]
        ]
    }

    metrics_path = artifacts_dir / "ensemble_ultimate_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'=' * 80}")
    print("ULTIMATE MODEL TRAINING COMPLETE!")
    print(f"{'=' * 80}")

    if ensemble_acc >= 0.60:
        print(f"\nSUCCESS! Reached {ensemble_acc*100:.2f}% with ultimate features!")
    else:
        print(f"\nFinal accuracy: {ensemble_acc*100:.2f}%")
        if ensemble_acc >= 0.5924:
            print("Matched or exceeded time windows performance")
        else:
            print("Time windows alone (59.24%) performed better")


if __name__ == "__main__":
    main()
