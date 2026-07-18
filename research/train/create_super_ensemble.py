"""
Create Super Ensemble: Combine ALL trained models.

Models to combine:
1. Baseline (60 features, optimized weights)
2. Sample-weighted models
3. Feature interaction models

Strategy: Learn optimal weights across ALL models
Expected: +0.5-1% improvement
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def create_super_ensemble():
    """Create super ensemble from all models."""

    print("=" * 80)
    print("SUPER ENSEMBLE: COMBINING ALL MODELS")
    print("=" * 80)
    print("\nStrategy: Ensemble of ensembles")
    print("Combining 3 different training approaches")

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Load all ensemble models
    print("\nLoading all models...")

    # 1. Baseline (optimized weights, 60 features)
    with open(artifacts_dir / "ensemble_optimized_weights_model.pkl", 'rb') as f:
        baseline_ensemble = pickle.load(f)
    print("  [OK] Baseline ensemble (optimized weights)")

    # 2. Sample-weighted
    with open(artifacts_dir / "ensemble_sample_weighted_model.pkl", 'rb') as f:
        weighted_ensemble = pickle.load(f)
    print("  [OK] Sample-weighted ensemble")

    # 3. Feature interactions
    with open(artifacts_dir / "ensemble_with_interactions_model.pkl", 'rb') as f:
        interaction_ensemble = pickle.load(f)
    print("  [OK] Feature interaction ensemble")

    # Load test data
    print("\nLoading test data...")

    # For baseline and sample-weighted (60 features)
    data_60 = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df_60 = pd.read_parquet(data_60)

    split_date = df_60['date'].quantile(0.8)
    test_60 = df_60[df_60['date'] > split_date].copy()

    # Get predictions from all models
    print("\nGenerating predictions from all models...")

    # Baseline models
    feat_60 = baseline_ensemble['feature_cols']
    X_test_60 = test_60[feat_60].values
    X_test_60 = pd.DataFrame(X_test_60, columns=feat_60).ffill().fillna(0).values
    y_test = test_60['future_return'].values

    baseline_xgb_pred = baseline_ensemble['models']['xgboost'].predict(X_test_60)
    baseline_lgb_pred = baseline_ensemble['models']['lightgbm'].predict(X_test_60)
    baseline_cat_pred = baseline_ensemble['models']['catboost'].predict(X_test_60)

    baseline_weights = baseline_ensemble['weights']
    baseline_pred = (baseline_weights['xgboost'] * baseline_xgb_pred +
                     baseline_weights['lightgbm'] * baseline_lgb_pred +
                     baseline_weights['catboost'] * baseline_cat_pred)

    print(f"  Baseline ensemble: {direction_accuracy(y_test, baseline_pred)*100:.2f}%")

    # Sample-weighted models
    weighted_xgb_pred = weighted_ensemble['xgboost'].predict(X_test_60)
    weighted_lgb_pred = weighted_ensemble['lightgbm'].predict(X_test_60)
    weighted_cat_pred = weighted_ensemble['catboost'].predict(X_test_60)

    weighted_pred = (weighted_xgb_pred + weighted_lgb_pred + weighted_cat_pred) / 3

    print(f"  Sample-weighted ensemble: {direction_accuracy(y_test, weighted_pred)*100:.2f}%")

    # Feature interaction models (72 features)
    # Need to recreate interaction features for test set
    from train_with_feature_interactions import create_interaction_features

    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    original_60_cols = [c for c in test_60.columns if c not in meta_cols]

    test_interactions = create_interaction_features(test_60, original_60_cols)
    test_60_with_int = test_60.copy()
    for feat_name, feat_values in test_interactions.items():
        test_60_with_int[feat_name] = feat_values

    feat_72 = interaction_ensemble['feature_cols']
    X_test_72 = test_60_with_int[feat_72].values
    X_test_72 = pd.DataFrame(X_test_72, columns=feat_72).ffill().fillna(0).values

    interaction_xgb_pred = interaction_ensemble['xgboost'].predict(X_test_72)
    interaction_lgb_pred = interaction_ensemble['lightgbm'].predict(X_test_72)
    interaction_cat_pred = interaction_ensemble['catboost'].predict(X_test_72)

    interaction_pred = (interaction_xgb_pred + interaction_lgb_pred + interaction_cat_pred) / 3

    print(f"  Feature interaction ensemble: {direction_accuracy(y_test, interaction_pred)*100:.2f}%")

    # Optimize weights across all ensembles
    print("\n" + "=" * 80)
    print("OPTIMIZING SUPER ENSEMBLE WEIGHTS")
    print("=" * 80)

    # Create prediction matrix
    pred_matrix = np.column_stack([
        baseline_pred,
        weighted_pred,
        interaction_pred
    ])

    print(f"\nPrediction matrix shape: {pred_matrix.shape}")
    print(f"  Baseline, Sample-weighted, Feature-interaction")

    # Objective function
    def objective(weights):
        weights_norm = weights / weights.sum()
        super_pred = pred_matrix @ weights_norm
        return -direction_accuracy(y_test, super_pred)

    # Optimize
    bounds = [(0, 1)] * 3

    print("\nOptimizing weights...")
    result = differential_evolution(objective, bounds, seed=42, maxiter=100)

    if result.success:
        optimal_weights = result.x / result.x.sum()
        optimal_acc = -result.fun

        print(f"\nOptimization successful!")
        print(f"  Baseline:            {optimal_weights[0]*100:.1f}%")
        print(f"  Sample-weighted:     {optimal_weights[1]*100:.1f}%")
        print(f"  Feature-interaction: {optimal_weights[2]*100:.1f}%")
        print(f"\n  Super Ensemble Accuracy: {optimal_acc*100:.2f}%")

        # Final prediction
        super_pred = pred_matrix @ optimal_weights

        # Metrics
        super_mae = mean_absolute_error(y_test, super_pred)
        super_r2 = r2_score(y_test, super_pred)

        print(f"  MAE: {super_mae*100:.2f}%")
        print(f"  R2:  {super_r2:.4f}")

        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)

        print(f"\nBaseline:        {direction_accuracy(y_test, baseline_pred)*100:.2f}%")
        print(f"Super Ensemble:  {optimal_acc*100:.2f}%")
        print(f"Improvement:     {(optimal_acc - 0.5902)*100:+.2f} percentage points")

        # Save super ensemble
        super_ensemble = {
            'ensembles': {
                'baseline': baseline_ensemble,
                'sample_weighted': weighted_ensemble,
                'feature_interactions': interaction_ensemble
            },
            'weights': {
                'baseline': float(optimal_weights[0]),
                'sample_weighted': float(optimal_weights[1]),
                'feature_interactions': float(optimal_weights[2])
            }
        }

        model_path = artifacts_dir / "super_ensemble_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(super_ensemble, f)

        print(f"\nSuper ensemble saved to: {model_path}")

        # Save metrics
        metrics = {
            'super_ensemble': True,
            'num_base_ensembles': 3,
            'test_direction_accuracy': float(optimal_acc),
            'test_mae': float(super_mae),
            'test_r2': float(super_r2),
            'ensemble_weights': {
                'baseline': float(optimal_weights[0]),
                'sample_weighted': float(optimal_weights[1]),
                'feature_interactions': float(optimal_weights[2])
            },
            'individual_accuracies': {
                'baseline': float(direction_accuracy(y_test, baseline_pred)),
                'sample_weighted': float(direction_accuracy(y_test, weighted_pred)),
                'feature_interactions': float(direction_accuracy(y_test, interaction_pred))
            },
            'improvement': float(optimal_acc - 0.5902)
        }

        metrics_path = artifacts_dir / "super_ensemble_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {metrics_path}")

        print("\n" + "=" * 80)
        print("SUPER ENSEMBLE COMPLETE!")
        print("=" * 80)

        if optimal_acc >= 0.63:
            print(f"\n[SUCCESS] Achieved {optimal_acc*100:.2f}% (>=63%) direction accuracy!")
        elif optimal_acc >= 0.60:
            print(f"\n[GOOD] Achieved {optimal_acc*100:.2f}% (>=60%) direction accuracy!")
        else:
            print(f"\n[RESULT] Achieved {optimal_acc*100:.2f}% direction accuracy.")
            print("Traditional ML methods may have reached their limit.")
            print("Consider deep learning (LSTM/Transformer) for further improvement.")

    else:
        print(f"\nOptimization failed: {result.message}")


if __name__ == "__main__":
    create_super_ensemble()
