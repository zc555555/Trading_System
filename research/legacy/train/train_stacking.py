"""
Train Stacking Ensemble with Meta-Learner for maximum direction accuracy.

Stacking uses a meta-model (Ridge Regression) to learn optimal weights
for combining XGBoost, LightGBM, and CatBoost predictions.

Expected improvement: 58% -> 60-61% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StackingEnsemble:
    """
    Stacking Ensemble with meta-learner.

    Level 0: Base models (XGBoost, LightGBM, CatBoost)
    Level 1: Meta-model (Ridge Regression) learns optimal weights
    """

    def __init__(self, base_models=None, meta_model=None):
        """
        Initialize stacking ensemble.

        Args:
            base_models: Dict of {model_name: model_object}
            meta_model: Meta-learner (Ridge by default)
        """
        self.base_models = base_models or {}
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train stacking ensemble.

        Step 1: Get predictions from base models on validation set
        Step 2: Train meta-model on these predictions

        Args:
            X_train: Training features (used by base models, already trained)
            y_train: Training labels (not used, base models already trained)
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "=" * 80)
        print("TRAINING STACKING ENSEMBLE")
        print("=" * 80)

        # Step 1: Get predictions from base models on validation set
        print("\nStep 1: Getting predictions from base models on validation set...")
        val_meta_features = np.zeros((len(X_val), len(self.base_models)))

        for i, (model_name, model) in enumerate(self.base_models.items()):
            print(f"  [{i+1}/{len(self.base_models)}] {model_name}...")
            val_meta_features[:, i] = model.predict(X_val)

        print(f"\nMeta-features shape: {val_meta_features.shape}")

        # Step 2: Train meta-model on validation predictions
        print("\nStep 2: Training meta-model (Ridge Regression) on validation set...")
        self.meta_model.fit(val_meta_features, y_val)

        # Print learned weights
        print("\nLearned meta-model weights:")
        for i, (model_name, _) in enumerate(self.base_models.items()):
            weight = self.meta_model.coef_[i]
            print(f"  {model_name:12s}: {weight:+.4f}")
        print(f"  Intercept:    {self.meta_model.intercept_:+.4f}")

        self.is_fitted = True

        # Evaluate on validation set
        print("\nValidation set performance:")
        val_pred = self.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        val_direction = ((y_val > 0) == (val_pred > 0)).mean()

        print(f"  MAE: {val_mae:.6f} ({val_mae*100:.2f}%)")
        print(f"  R2:  {val_r2:.4f}")
        print(f"  Direction Accuracy: {val_direction*100:.2f}%")

        return self

    def predict(self, X):
        """
        Make predictions using stacking ensemble.

        Args:
            X: Feature matrix

        Returns:
            Stacked predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Step 1: Get predictions from all base models
        meta_features = np.zeros((len(X), len(self.base_models)))

        for i, (model_name, model) in enumerate(self.base_models.items()):
            meta_features[:, i] = model.predict(X)

        # Step 2: Use meta-model to combine predictions
        stacked_pred = self.meta_model.predict(meta_features)

        return stacked_pred

    def get_individual_predictions(self, X):
        """Get predictions from each base model."""
        return {
            model_name: model.predict(X)
            for model_name, model in self.base_models.items()
        }


def train_stacking():
    """Train stacking ensemble and evaluate performance."""

    print("=" * 80)
    print("STACKING ENSEMBLE TRAINING")
    print("=" * 80)
    print("\nStacking learns optimal weights to combine base models.")
    print("Expected: 58% -> 60-61% direction accuracy")

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Load individual models
    print("\n" + "=" * 80)
    print("LOADING BASE MODELS")
    print("=" * 80)

    base_models = {}

    # XGBoost
    xgb_path = artifacts_dir / "xgboost_regression_model.pkl"
    if xgb_path.exists():
        with open(xgb_path, 'rb') as f:
            base_models['xgboost'] = pickle.load(f)
        print("  [OK] XGBoost loaded")
    else:
        print(f"  [ERROR] XGBoost not found at {xgb_path}")
        print("\nPlease train base models first:")
        print("  python train/train_xgb_regression.py")
        print("  python train/train_lightgbm_regression.py")
        print("  python train/train_catboost_regression.py")
        return

    # LightGBM
    lgb_path = artifacts_dir / "lightgbm_regression_model.pkl"
    if lgb_path.exists():
        with open(lgb_path, 'rb') as f:
            base_models['lightgbm'] = pickle.load(f)
        print("  [OK] LightGBM loaded")
    else:
        print(f"  [WARN] LightGBM not found at {lgb_path}")

    # CatBoost
    cb_path = artifacts_dir / "catboost_regression_model.pkl"
    if cb_path.exists():
        with open(cb_path, 'rb') as f:
            base_models['catboost'] = pickle.load(f)
        print("  [OK] CatBoost loaded")
    else:
        print(f"  [WARN] CatBoost not found at {cb_path}")

    if len(base_models) < 2:
        print("\nERROR: Need at least 2 base models for stacking!")
        return

    print(f"\nLoaded {len(base_models)} base models: {list(base_models.keys())}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    parquet_path = Path(__file__).parent.parent / "data" / "stocks.parquet"
    if not parquet_path.exists():
        print(f"ERROR: Dataset not found at {parquet_path}")
        print("Please run: python run_build_fixed.py")
        return

    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Dataset shape: {df.shape}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Features: {len(feature_cols)}")

    # Split data (70/10/20 time-based: train/val/test)
    train_split = df['date'].quantile(0.7)
    val_split = df['date'].quantile(0.8)

    train_df = df[df['date'] <= train_split].copy()
    val_df = df[(df['date'] > train_split) & (df['date'] <= val_split)].copy()
    test_df = df[df['date'] > val_split].copy()

    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['future_return'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['future_return'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_val = pd.DataFrame(X_val, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create and train stacking ensemble
    print("\n" + "=" * 80)
    print("TRAINING STACKING ENSEMBLE")
    print("=" * 80)

    stacking = StackingEnsemble(
        base_models=base_models,
        meta_model=Ridge(alpha=1.0, random_state=42)
    )

    # Train meta-model on validation set
    stacking.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Get predictions
    stacking_pred = stacking.predict(X_test)
    individual_preds = stacking.get_individual_predictions(X_test)

    # Stacking metrics
    stacking_mse = mean_squared_error(y_test, stacking_pred)
    stacking_mae = mean_absolute_error(y_test, stacking_pred)
    stacking_r2 = r2_score(y_test, stacking_pred)
    stacking_direction = ((y_test > 0) == (stacking_pred > 0)).mean()

    print(f"\nSTACKING ENSEMBLE:")
    print(f"  MSE: {stacking_mse:.6f}")
    print(f"  MAE: {stacking_mae:.6f} ({stacking_mae*100:.2f}%)")
    print(f"  R2:  {stacking_r2:.4f}")
    print(f"  Direction Accuracy: {stacking_direction*100:.2f}%")

    # Compare with individual models
    print(f"\n" + "=" * 80)
    print("COMPARISON: STACKING VS BASE MODELS")
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

    # Compare with simple weighted average
    print(f"\nSIMPLE WEIGHTED AVERAGE (35% + 35% + 30%):")
    weights = config['model']['ensemble']['weights']

    weighted_pred = np.zeros(len(X_test))
    for model_name, pred in individual_preds.items():
        weight = weights.get(model_name, 1.0 / len(individual_preds))
        weighted_pred += weight * pred

    weighted_mae = mean_absolute_error(y_test, weighted_pred)
    weighted_r2 = r2_score(y_test, weighted_pred)
    weighted_direction = ((y_test > 0) == (weighted_pred > 0)).mean()

    print(f"  MAE: {weighted_mae:.6f} ({weighted_mae*100:.2f}%)")
    print(f"  R2:  {weighted_r2:.4f}")
    print(f"  Direction: {weighted_direction*100:.2f}%")

    # Final comparison
    print(f"\n" + "=" * 80)
    print("STACKING IMPROVEMENT")
    print("=" * 80)

    best_individual_mae = min([c['mae'] for c in comparison])
    best_individual_direction = max([c['direction_acc'] for c in comparison])

    mae_improvement = (best_individual_mae - stacking_mae) / best_individual_mae * 100
    direction_improvement = (stacking_direction - weighted_direction) * 100

    print(f"\nStacking vs Best Individual Model:")
    print(f"  MAE Improvement: {mae_improvement:+.2f}%")
    print(f"  Direction Improvement: {direction_improvement:+.2f} percentage points")

    print(f"\nStacking vs Weighted Average:")
    print(f"  Direction: {weighted_direction*100:.2f}% -> {stacking_direction*100:.2f}%")
    print(f"  Improvement: {direction_improvement:+.2f} percentage points")

    # Save stacking model
    stacking_path = artifacts_dir / "stacking_regression_model.pkl"
    with open(stacking_path, 'wb') as f:
        pickle.dump(stacking, f)

    print(f"\n" + "=" * 80)
    print(f"Model saved to: {stacking_path}")
    print("=" * 80)

    # Save metrics
    metrics = {
        "model_type": "Stacking Ensemble (XGBoost + LightGBM + CatBoost + Ridge)",
        "stacking": {
            "test_mse": float(stacking_mse),
            "test_mae": float(stacking_mae),
            "test_r2": float(stacking_r2),
            "test_direction_accuracy": float(stacking_direction)
        },
        "base_models": comparison,
        "simple_weighted_average": {
            "test_mae": float(weighted_mae),
            "test_r2": float(weighted_r2),
            "test_direction_accuracy": float(weighted_direction)
        },
        "meta_model_weights": {
            model_name: float(stacking.meta_model.coef_[i])
            for i, model_name in enumerate(base_models.keys())
        },
        "meta_model_intercept": float(stacking.meta_model.intercept_),
        "improvement": {
            "mae_improvement_pct": float(mae_improvement),
            "direction_improvement_points": float(direction_improvement)
        }
    }

    metrics_path = artifacts_dir / "stacking_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("STACKING TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\nFinal Results:")
    print(f"  Direction Accuracy: {stacking_direction*100:.2f}%")
    print(f"  MAE: {stacking_mae*100:.2f}%")
    print(f"  R²: {stacking_r2:.4f}")

    if stacking_direction >= 0.60:
        print(f"\n[SUCCESS] Achieved {stacking_direction*100:.2f}% (>60%) direction accuracy!")
    elif stacking_direction >= 0.595:
        print(f"\n[CLOSE] Achieved {stacking_direction*100:.2f}% direction accuracy.")
        print("   Try adding directional features for further improvement.")
    else:
        print(f"\n[NOTE] Achieved {stacking_direction*100:.2f}% direction accuracy.")
        print("   Recommend trying additional improvements:")
        print("   1. Feature selection (remove noisy features)")
        print("   2. Add directional features")
        print("   3. Hyperparameter optimization")

    print("\nTo use stacking model in predictions:")
    print("  Update get_stock_detail.py to load stacking_regression_model.pkl")


if __name__ == "__main__":
    train_stacking()
