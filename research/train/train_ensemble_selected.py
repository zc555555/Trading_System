"""
Create Ensemble Model from SELECTED FEATURES models.

Combines XGBoost, LightGBM, and CatBoost trained on 60 selected features.
Expected improvement: 58.42% -> 59-60% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_ensemble_selected():
    """Create ensemble from selected features models."""

    print("=" * 80)
    print("CREATING ENSEMBLE MODEL (SELECTED 60 FEATURES)")
    print("=" * 80)

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Load individual models
    print("\nLoading individual models...")

    models = {}

    # XGBoost
    xgb_path = artifacts_dir / "xgboost_selected_model.pkl"
    with open(xgb_path, 'rb') as f:
        models['xgboost'] = pickle.load(f)
    print("  [OK] XGBoost loaded")

    # LightGBM
    lgb_path = artifacts_dir / "lightgbm_selected_model.pkl"
    with open(lgb_path, 'rb') as f:
        models['lightgbm'] = pickle.load(f)
    print("  [OK] LightGBM loaded")

    # CatBoost
    cb_path = artifacts_dir / "catboost_selected_model.pkl"
    with open(cb_path, 'rb') as f:
        models['catboost'] = pickle.load(f)
    print("  [OK] CatBoost loaded")

    print(f"\nLoaded {len(models)} models: {list(models.keys())}")

    # Ensemble weights (will be optimized in next step)
    weights = {'xgboost': 0.35, 'lightgbm': 0.35, 'catboost': 0.3}
    print(f"\nEnsemble weights: {weights}")

    # Load test dataset
    print("\nLoading test dataset...")
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Get predictions from each model
    print("\nGenerating predictions from each model...")

    individual_preds = {}
    for model_name, model in models.items():
        individual_preds[model_name] = model.predict(X_test)
        print(f"  [OK] {model_name}")

    # Weighted average ensemble
    ensemble_pred = np.zeros(len(X_test))
    for model_name, pred in individual_preds.items():
        weight = weights[model_name]
        ensemble_pred += weight * pred

    # Evaluate
    print("\n" + "=" * 80)
    print("ENSEMBLE PERFORMANCE")
    print("=" * 80)

    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_direction = ((y_test > 0) == (ensemble_pred > 0)).mean()

    print(f"\nEnsemble Results:")
    print(f"  MSE: {ensemble_mse:.6f}")
    print(f"  MAE: {ensemble_mae:.6f} ({ensemble_mae*100:.2f}%)")
    print(f"  R2:  {ensemble_r2:.4f}")
    print(f"  Direction Accuracy: {ensemble_direction*100:.2f}%")

    # Compare with individual models
    print("\n" + "=" * 80)
    print("COMPARISON: ENSEMBLE VS INDIVIDUAL MODELS")
    print("=" * 80)

    comparison = []
    for model_name, pred in individual_preds.items():
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        direction = ((y_test > 0) == (pred > 0)).mean()

        comparison.append({
            'model': model_name,
            'mae': float(mae),
            'r2': float(r2),
            'direction_acc': float(direction)
        })

        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {mae:.6f} ({mae*100:.2f}%)")
        print(f"  R2:  {r2:.4f}")
        print(f"  Direction: {direction*100:.2f}%")

    print(f"\nENSEMBLE (Weighted Average):")
    print(f"  MAE: {ensemble_mae:.6f} ({ensemble_mae*100:.2f}%)")
    print(f"  R2:  {ensemble_r2:.4f}")
    print(f"  Direction: {ensemble_direction*100:.2f}%")

    # Improvement analysis
    best_individual_mae = min([c['mae'] for c in comparison])
    best_individual_direction = max([c['direction_acc'] for c in comparison])

    mae_improvement = (best_individual_mae - ensemble_mae) / best_individual_mae * 100
    direction_improvement = (ensemble_direction - best_individual_direction) * 100

    print("\n" + "=" * 80)
    print(f"Ensemble Improvement: {direction_improvement:+.2f} percentage points")
    print("=" * 80)

    # Save ensemble model
    ensemble_model = {
        'models': models,
        'weights': weights,
        'feature_cols': feature_cols
    }

    model_path = artifacts_dir / "ensemble_selected_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_model, f)

    print(f"\nEnsemble model saved to: {model_path}")

    # Save metrics
    metrics = {
        "model_type": "Ensemble (Selected 60 Features)",
        "features_count": len(feature_cols),
        "ensemble": {
            "test_mse": float(ensemble_mse),
            "test_mae": float(ensemble_mae),
            "test_r2": float(ensemble_r2),
            "test_direction_accuracy": float(ensemble_direction)
        },
        "individual_models": comparison,
        "weights": weights,
        "improvement": {
            "mae_improvement_pct": float(mae_improvement),
            "direction_improvement_points": float(direction_improvement)
        }
    }

    metrics_path = artifacts_dir / "ensemble_selected_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("ENSEMBLE CREATION COMPLETE!")
    print("=" * 80)

    print(f"\nFinal Test MAE: {ensemble_mae*100:.2f}%")
    print(f"Final Test R2:  {ensemble_r2:.4f}")
    print(f"Direction Accuracy: {ensemble_direction*100:.2f}%")

    print("\nComparison with 175 features:")
    print(f"  175 features: 58.42% direction accuracy")
    print(f"  60 features:  {ensemble_direction*100:.2f}% direction accuracy")
    improvement_vs_175 = (ensemble_direction - 0.5842) * 100
    print(f"  Improvement:  {improvement_vs_175:+.2f} percentage points")

    if ensemble_direction >= 0.60:
        print(f"\n[SUCCESS] Achieved {ensemble_direction*100:.2f}% (>60%) direction accuracy!")
    elif ensemble_direction >= 0.595:
        print(f"\n[CLOSE] Achieved {ensemble_direction*100:.2f}% direction accuracy.")
        print("  Next step: Hyperparameter optimization")
    else:
        print(f"\n[IN PROGRESS] Achieved {ensemble_direction*100:.2f}% direction accuracy.")
        print("  Next steps:")
        print("    1. Hyperparameter optimization")
        print("    2. Optimize ensemble weights")

    print("\nNext step: Run hyperparameter optimization")
    print("  python train/optimize_hyperparams.py")


if __name__ == "__main__":
    create_ensemble_selected()
