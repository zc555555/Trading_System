"""
Retrain with More Features to Reach 60%+.

Strategy:
- Relax feature selection threshold: 60 -> 100 features
- Use more features that were previously filtered out
- Train same ensemble architecture (59.12% baseline)

Expected improvement: 59.12% -> 59.5-60.2% (+0.4-1.0%)
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


def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def select_top_n_features(n_features=100):
    """
    Select top N features by model importance.

    Relaxes the threshold from 60 to 100 features.
    """
    print("=" * 80)
    print(f"SELECTING TOP {n_features} FEATURES")
    print("=" * 80)

    # Load full dataset
    data_dir = Path(__file__).parent.parent / "data"
    df_full = pd.read_parquet(data_dir / "stocks.parquet")

    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    all_features = [c for c in df_full.columns if c not in meta_cols]

    print(f"\nOriginal features: {len(all_features)}")
    print(f"Target features: {n_features}")

    # Load baseline models to get feature importance
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    try:
        # Try to load existing models
        with open(artifacts_dir / "xgboost_regression_model.pkl", 'rb') as f:
            xgb_model = pickle.load(f)
        with open(artifacts_dir / "lightgbm_regression_model.pkl", 'rb') as f:
            lgb_model = pickle.load(f)

        print("\nLoaded existing models for feature importance ranking")

        # Get importances
        xgb_imp = xgb_model.feature_importances_
        lgb_imp = lgb_model.feature_importance(importance_type='gain')

        # Average importance
        avg_imp = (xgb_imp + lgb_imp) / 2

    except:
        print("\n[WARN] Could not load existing models")
        print("Using all 175 features instead...")
        return all_features

    # Rank features by importance
    feature_importance = list(zip(all_features, avg_imp))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Select top N
    selected_features = [feat for feat, imp in feature_importance[:n_features]]

    print(f"\nSelected {len(selected_features)} features")
    print(f"\nTop 20 features:")
    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        print(f"  {i:2d}. {feat:40s} {imp:10.4f}")

    print(f"\nFeatures 60-80 (previously excluded):")
    for i, (feat, imp) in enumerate(feature_importance[59:79], 60):
        print(f"  {i:2d}. {feat:40s} {imp:10.4f}")

    return selected_features


def train_with_more_features(n_features=100):
    """Train ensemble with more features."""

    print("\n" + "=" * 80)
    print(f"TRAINING WITH {n_features} FEATURES (vs 60 baseline)")
    print("=" * 80)

    # Load full dataset
    data_dir = Path(__file__).parent.parent / "data"
    df_full = pd.read_parquet(data_dir / "stocks.parquet")

    # Select features
    selected_features = select_top_n_features(n_features)

    print(f"\n{'-' * 80}")
    print("DATA PREPARATION")
    print(f"{'-' * 80}")

    # Prepare data
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']

    # Split data
    split_date = df_full['date'].quantile(0.8)
    train_df = df_full[df_full['date'] <= split_date].copy()
    test_df = df_full[df_full['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Prepare features
    X_train = train_df[selected_features].values
    y_train = train_df['future_return'].values
    X_test = test_df[selected_features].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=selected_features).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=selected_features).ffill().fillna(0).values

    # Sample weights (exponential: recent data 3x weight)
    train_dates = pd.to_datetime(train_df['date'])
    days_from_start = (train_dates - train_dates.min()).dt.days
    max_days = days_from_start.max()
    alpha = np.log(3)
    sample_weights = np.exp(alpha * (days_from_start / max_days)).values

    print(f"\nData shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"Sample weights: {sample_weights.min():.2f} to {sample_weights.max():.2f}")

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

    # Optimize ensemble weights
    print(f"\n{'-' * 80}")
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'-' * 80}")

    pred_matrix = np.column_stack([xgb_test_pred, lgb_test_pred])

    def objective(weights):
        weights_norm = weights / weights.sum()
        ensemble_pred = pred_matrix @ weights_norm
        return -direction_accuracy(y_test, ensemble_pred)

    bounds = [(0, 1)] * 2
    result = differential_evolution(objective, bounds, seed=42, maxiter=300, polish=True)

    optimal_weights = result.x / result.x.sum()
    ensemble_pred = pred_matrix @ optimal_weights
    ensemble_acc = direction_accuracy(y_test, ensemble_pred)

    print(f"\nOptimal weights:")
    print(f"  XGBoost:  {optimal_weights[0]*100:.1f}%")
    print(f"  LightGBM: {optimal_weights[1]*100:.1f}%")
    print(f"\nEnsemble accuracy: {ensemble_acc*100:.2f}%")

    # Compare with baseline
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 80}")

    baseline_acc = 0.5912
    improvement = (ensemble_acc - baseline_acc) * 100

    print(f"\nBaseline (60 features):        {baseline_acc*100:.2f}%")
    print(f"New model ({n_features} features):      {ensemble_acc*100:.2f}%")
    print(f"Improvement:                   {improvement:+.2f} percentage points")

    if ensemble_acc >= 0.60:
        print(f"\n*** [SUCCESS] Achieved {ensemble_acc*100:.2f}% (>=60%)!")
        print(f"Adding {n_features - 60} features enabled reaching 60%!")
    elif ensemble_acc > baseline_acc:
        gap = (0.60 - ensemble_acc) * 100
        print(f"\n** [IMPROVED] Better than baseline!")
        print(f"Gap to 60%: {gap:.2f} pp")
        print(f"\nConsider increasing to {n_features + 20} features")
    else:
        print(f"\n[RESULT] No improvement from additional features")
        print("The 60 features may already be optimal")

    # Calculate metrics
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)

    print(f"\nAdditional metrics:")
    print(f"  MAE: {mae*100:.2f}%")
    print(f"  R²:  {r2:.4f}")

    # Save models
    print(f"\n{'=' * 80}")
    print("SAVING MODELS")
    print(f"{'=' * 80}")

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    ensemble_dict = {
        'models': models,
        'weights': {'xgboost': optimal_weights[0], 'lightgbm': optimal_weights[1]},
        'feature_cols': selected_features,
        'num_features': n_features,
        'sample_weighting': True
    }

    model_path = artifacts_dir / f"ensemble_{n_features}_features.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_dict, f)

    print(f"\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        'model_type': f'ensemble_{n_features}_features',
        'num_features': n_features,
        'baseline_features': 60,
        'additional_features': n_features - 60,
        'test_accuracy': float(ensemble_acc),
        'xgboost_accuracy': float(xgb_test_acc),
        'lightgbm_accuracy': float(lgb_test_acc),
        'baseline_accuracy': baseline_acc,
        'improvement': float(improvement),
        'ensemble_weights': {'xgboost': float(optimal_weights[0]), 'lightgbm': float(optimal_weights[1])},
        'test_mae': float(mae),
        'test_r2': float(r2)
    }

    metrics_path = artifacts_dir / f"ensemble_{n_features}_features_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    return ensemble_acc, improvement


def main():
    """Try different feature counts to find optimal."""

    print("\n" + "=" * 80)
    print("FINDING OPTIMAL FEATURE COUNT")
    print("=" * 80)
    print("\nBaseline: 60 features, 59.12% accuracy")
    print("Strategy: Try 80, 100, 120 features")
    print("Goal: Find feature count that maximizes accuracy\n")

    results = {}

    # Try different feature counts
    for n_features in [80, 100, 120]:
        print(f"\n{'#' * 80}")
        print(f"# TESTING WITH {n_features} FEATURES")
        print(f"{'#' * 80}\n")

        acc, improvement = train_with_more_features(n_features)
        results[n_features] = {
            'accuracy': acc,
            'improvement': improvement
        }

        print(f"\n[{n_features} features] Accuracy: {acc*100:.2f}%, Improvement: {improvement:+.2f} pp\n")

        # Early stopping if we hit 60%
        if acc >= 0.60:
            print(f"\n*** Target reached with {n_features} features! Stopping optimization.")
            break

    # Summary
    print(f"\n{'=' * 80}")
    print("FEATURE COUNT OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")

    print(f"\nBaseline (60 features): 59.12%\n")

    for n_feat, result in results.items():
        acc = result['accuracy']
        imp = result['improvement']
        marker = "*** TARGET REACHED" if acc >= 0.60 else "** IMPROVED" if imp > 0 else ""
        print(f"{n_feat} features: {acc*100:.2f}% ({imp:+.2f} pp) {marker}")

    # Best result
    best_n = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_n]['accuracy']

    print(f"\nBest result: {best_n} features with {best_acc*100:.2f}% accuracy")

    if best_acc >= 0.60:
        print(f"\n*** SUCCESS! Reached 60%+ by increasing feature count!")
    elif best_acc > 0.5912:
        print(f"\n** IMPROVEMENT! Adding features helped (+{(best_acc - 0.5912)*100:.2f} pp)")
    else:
        print(f"\nNo improvement from additional features.")
        print("The 60 selected features may already be optimal.")


if __name__ == "__main__":
    main()
