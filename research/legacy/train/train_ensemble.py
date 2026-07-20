"""
Create ensemble of XGBoost + LightGBM + CatBoost regression models.
Combines predictions from all three models for maximum accuracy.
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


class EnsembleRegressor:
    """Ensemble of multiple regression models with weighted averaging."""

    def __init__(self, models=None, weights=None):
        """
        Initialize ensemble.

        Args:
            models: Dict of {model_name: model_object}
            weights: Dict of {model_name: weight} (should sum to 1.0)
        """
        self.models = models or {}
        self.weights = weights or {}

        # Normalize weights
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def predict(self, X):
        """
        Make ensemble predictions.

        Args:
            X: Feature matrix

        Returns:
            Weighted average predictions
        """
        if not self.models:
            raise ValueError("No models loaded")

        predictions = []
        weights_list = []

        for model_name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights.get(model_name, 1.0 / len(self.models))

            predictions.append(pred)
            weights_list.append(weight)

        # Weighted average
        predictions = np.array(predictions)
        weights_array = np.array(weights_list).reshape(-1, 1)

        ensemble_pred = np.sum(predictions * weights_array, axis=0)

        return ensemble_pred

    def get_individual_predictions(self, X):
        """Get predictions from each individual model."""
        return {
            model_name: model.predict(X)
            for model_name, model in self.models.items()
        }


def train_ensemble():
    """Create and evaluate ensemble model."""

    print("=" * 80)
    print("CREATING ENSEMBLE MODEL (XGBoost + LightGBM + CatBoost)")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Load individual models
    print("\nLoading individual models...")
    models = {}

    # XGBoost
    xgb_path = artifacts_dir / "xgboost_regression_model.pkl"
    if xgb_path.exists():
        with open(xgb_path, 'rb') as f:
            models['xgboost'] = pickle.load(f)
        print("  [OK] XGBoost loaded")
    else:
        print(f"  [WARN] XGBoost not found at {xgb_path}")

    # LightGBM
    lgb_path = artifacts_dir / "lightgbm_regression_model.pkl"
    if lgb_path.exists():
        with open(lgb_path, 'rb') as f:
            models['lightgbm'] = pickle.load(f)
        print("  [OK] LightGBM loaded")
    else:
        print(f"  [WARN] LightGBM not found at {lgb_path}")

    # CatBoost
    cb_path = artifacts_dir / "catboost_regression_model.pkl"
    if cb_path.exists():
        with open(cb_path, 'rb') as f:
            models['catboost'] = pickle.load(f)
        print("  [OK] CatBoost loaded")
    else:
        print(f"  [WARN] CatBoost not found at {cb_path}")

    if len(models) == 0:
        print("\nERROR: No models found!")
        print("Please train individual models first:")
        print("  python train/train_xgb_regression.py")
        print("  python train/train_lightgbm_regression.py")
        print("  python train/train_catboost_regression.py")
        return

    print(f"\nLoaded {len(models)} models: {list(models.keys())}")

    # Get weights from config
    weights = config['model']['ensemble']['weights']
    print(f"\nEnsemble weights: {weights}")

    # Create ensemble
    ensemble = EnsembleRegressor(models=models, weights=weights)

    # Load test data for evaluation
    print("\nLoading test dataset...")
    parquet_path = Path(__file__).parent.parent / "data" / "stocks.parquet"
    df = pd.read_parquet(parquet_path)

    # Split
    split_date = df['date'].quantile(0.8)
    test_df = df[df['date'] > split_date].copy()

    # Get features
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Evaluate ensemble
    print("\n" + "=" * 80)
    print("ENSEMBLE PERFORMANCE")
    print("=" * 80)

    # Get predictions from ensemble and individual models
    ensemble_pred = ensemble.predict(X_test)
    individual_preds = ensemble.get_individual_predictions(X_test)

    # Ensemble metrics
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
    print(f"\n" + "=" * 80)
    print("COMPARISON: ENSEMBLE VS INDIVIDUAL MODELS")
    print("=" * 80)

    comparison = []
    for model_name, pred in individual_preds.items():
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        direction = ((y_test > 0) == (pred > 0)).mean()

        comparison.append({
            'model': model_name,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction
        })

        print(f"\n{model_name.upper()}:")
        print(f"  MAE: {mae:.6f} ({mae*100:.2f}%)")
        print(f"  R2:  {r2:.4f}")
        print(f"  Direction: {direction*100:.2f}%")

    print(f"\nENSEMBLE (Weighted Average):")
    print(f"  MAE: {ensemble_mae:.6f} ({ensemble_mae*100:.2f}%)")
    print(f"  R2:  {ensemble_r2:.4f}")
    print(f"  Direction: {ensemble_direction*100:.2f}%")

    # Calculate improvement
    best_individual_mae = min([c['mae'] for c in comparison])
    improvement = (best_individual_mae - ensemble_mae) / best_individual_mae * 100

    print(f"\n" + "=" * 80)
    print(f"Ensemble Improvement: {improvement:+.2f}% better MAE than best individual model")
    print("=" * 80)

    # Save ensemble
    ensemble_path = artifacts_dir / "ensemble_regression_model.pkl"
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble, f)

    print(f"\nEnsemble model saved to: {ensemble_path}")

    # Save metrics
    metrics = {
        "model_type": "Ensemble (XGBoost + LightGBM + CatBoost)",
        "ensemble": {
            "test_mse": float(ensemble_mse),
            "test_mae": float(ensemble_mae),
            "test_r2": float(ensemble_r2),
            "test_direction_accuracy": float(ensemble_direction)
        },
        "individual_models": comparison,
        "weights": weights,
        "improvement_over_best": float(improvement)
    }

    metrics_path = artifacts_dir / "ensemble_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("ENSEMBLE CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Test MAE: {ensemble_mae*100:.2f}%")
    print(f"Final Test R2:  {ensemble_r2:.4f}")
    print(f"Direction Accuracy: {ensemble_direction*100:.2f}%")
    print("\nThe ensemble model combines the strengths of all three models.")
    print("It should provide more accurate and stable predictions than any single model.")


if __name__ == "__main__":
    train_ensemble()
