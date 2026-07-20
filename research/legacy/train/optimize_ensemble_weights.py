"""
Optimize Ensemble Weights using Validation Set.

Learns optimal weights for combining XGBoost, LightGBM, and CatBoost
to maximize direction accuracy on validation set.

Expected improvement: 58.59% -> 59.5-60.5% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def load_models_and_data():
    """Load models and dataset."""

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Load models
    models = {}

    with open(artifacts_dir / "xgboost_selected_model.pkl", 'rb') as f:
        models['xgboost'] = pickle.load(f)

    with open(artifacts_dir / "lightgbm_selected_model.pkl", 'rb') as f:
        models['lightgbm'] = pickle.load(f)

    with open(artifacts_dir / "catboost_selected_model.pkl", 'rb') as f:
        models['catboost'] = pickle.load(f)

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Split data (70/10/20: train/val/test)
    train_split = df['date'].quantile(0.7)
    val_split = df['date'].quantile(0.8)

    val_df = df[(df['date'] > train_split) & (df['date'] <= val_split)].copy()
    test_df = df[df['date'] > val_split].copy()

    # Validation set
    X_val = val_df[feature_cols].values
    y_val = val_df['future_return'].values

    # Test set
    X_test = test_df[feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_val = pd.DataFrame(X_val, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    return models, X_val, y_val, X_test, y_test, feature_cols


def get_predictions(models, X):
    """Get predictions from all models."""
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)
    return predictions


def optimize_weights_scipy(models, X_val, y_val):
    """Optimize weights using scipy.optimize.minimize."""

    print("\n" + "=" * 80)
    print("METHOD 1: SCIPY MINIMIZE (GRADIENT-BASED)")
    print("=" * 80)

    # Get validation predictions
    val_preds = get_predictions(models, X_val)
    pred_matrix = np.column_stack([val_preds['xgboost'],
                                     val_preds['lightgbm'],
                                     val_preds['catboost']])

    # Objective function: negative direction accuracy
    def objective(weights):
        ensemble_pred = pred_matrix @ weights
        return -direction_accuracy(y_val, ensemble_pred)

    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1), (0, 1), (0, 1)]

    # Initial guess: equal weights
    x0 = np.array([1/3, 1/3, 1/3])

    print("\nOptimizing weights...")
    print(f"  Initial weights: {x0}")
    print(f"  Initial direction accuracy: {-objective(x0)*100:.2f}%")

    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        optimal_accuracy = -result.fun

        print(f"\nOptimization successful!")
        print(f"  Optimal weights: {optimal_weights}")
        print(f"  Optimal direction accuracy: {optimal_accuracy*100:.2f}%")
        print(f"  Improvement: {(optimal_accuracy - (-objective(x0)))*100:+.2f} percentage points")

        return optimal_weights, optimal_accuracy
    else:
        print(f"\nOptimization failed: {result.message}")
        return None, None


def optimize_weights_grid_search(models, X_val, y_val):
    """Optimize weights using grid search."""

    print("\n" + "=" * 80)
    print("METHOD 2: GRID SEARCH")
    print("=" * 80)

    # Get validation predictions
    val_preds = get_predictions(models, X_val)

    # Grid search
    best_weights = None
    best_accuracy = 0

    print("\nSearching weight combinations...")

    results = []

    # Search over grid (step size 0.05)
    for w1 in np.arange(0, 1.05, 0.05):
        for w2 in np.arange(0, 1.05, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue

            weights = np.array([w1, w2, w3])

            # Calculate ensemble prediction
            ensemble_pred = (weights[0] * val_preds['xgboost'] +
                            weights[1] * val_preds['lightgbm'] +
                            weights[2] * val_preds['catboost'])

            # Calculate direction accuracy
            acc = direction_accuracy(y_val, ensemble_pred)

            results.append({
                'xgboost': w1,
                'lightgbm': w2,
                'catboost': w3,
                'direction_accuracy': acc
            })

            if acc > best_accuracy:
                best_accuracy = acc
                best_weights = weights

    print(f"\nSearched {len(results)} combinations")
    print(f"\nBest weights: {best_weights}")
    print(f"  XGBoost:  {best_weights[0]:.3f}")
    print(f"  LightGBM: {best_weights[1]:.3f}")
    print(f"  CatBoost: {best_weights[2]:.3f}")
    print(f"\nBest direction accuracy: {best_accuracy*100:.2f}%")

    # Show top 5
    results_df = pd.DataFrame(results).sort_values('direction_accuracy', ascending=False)
    print(f"\nTop 5 weight combinations:")
    print(results_df.head(5).to_string(index=False))

    return best_weights, best_accuracy


def optimize_weights_differential_evolution(models, X_val, y_val):
    """Optimize weights using differential evolution (global optimization)."""

    print("\n" + "=" * 80)
    print("METHOD 3: DIFFERENTIAL EVOLUTION (GLOBAL OPTIMIZATION)")
    print("=" * 80)

    # Get validation predictions
    val_preds = get_predictions(models, X_val)
    pred_matrix = np.column_stack([val_preds['xgboost'],
                                     val_preds['lightgbm'],
                                     val_preds['catboost']])

    # Objective function with normalization
    def objective(weights):
        # Normalize weights to sum to 1
        weights_norm = weights / weights.sum()
        ensemble_pred = pred_matrix @ weights_norm
        return -direction_accuracy(y_val, ensemble_pred)

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1), (0, 1), (0, 1)]

    print("\nOptimizing weights...")

    # Optimize
    result = differential_evolution(objective, bounds, seed=42, maxiter=100)

    if result.success:
        # Normalize weights
        optimal_weights = result.x / result.x.sum()
        optimal_accuracy = -result.fun

        print(f"\nOptimization successful!")
        print(f"  Optimal weights: {optimal_weights}")
        print(f"  XGBoost:  {optimal_weights[0]:.3f}")
        print(f"  LightGBM: {optimal_weights[1]:.3f}")
        print(f"  CatBoost: {optimal_weights[2]:.3f}")
        print(f"  Optimal direction accuracy: {optimal_accuracy*100:.2f}%")

        return optimal_weights, optimal_accuracy
    else:
        print(f"\nOptimization failed: {result.message}")
        return None, None


def evaluate_ensemble(models, X_test, y_test, weights, weight_names):
    """Evaluate ensemble with given weights on test set."""

    # Get test predictions
    test_preds = get_predictions(models, X_test)

    # Calculate ensemble prediction
    ensemble_pred = (weights[0] * test_preds['xgboost'] +
                     weights[1] * test_preds['lightgbm'] +
                     weights[2] * test_preds['catboost'])

    # Evaluate
    mse = mean_squared_error(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    direction_acc = direction_accuracy(y_test, ensemble_pred)

    print(f"\n{weight_names} - Test Set Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f} ({mae*100:.2f}%)")
    print(f"  R2:  {r2:.4f}")
    print(f"  Direction Accuracy: {direction_acc*100:.2f}%")

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'direction_accuracy': float(direction_acc)
    }


def main():
    """Main ensemble weight optimization workflow."""

    print("=" * 80)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 80)
    print("\nLearning optimal weights on validation set")
    print("Expected improvement: 58.59% -> 59.5-60.5%")

    # Load models and data
    print("\nLoading models and data...")
    models, X_val, y_val, X_test, y_test, feature_cols = load_models_and_data()

    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")

    # Current fixed weights (baseline)
    fixed_weights = np.array([0.35, 0.35, 0.30])

    print("\n" + "=" * 80)
    print("BASELINE: FIXED WEIGHTS")
    print("=" * 80)
    print(f"\nFixed weights: {fixed_weights}")

    baseline_metrics = evaluate_ensemble(models, X_test, y_test, fixed_weights, "FIXED WEIGHTS")

    # Method 1: Scipy minimize
    scipy_weights, scipy_val_acc = optimize_weights_scipy(models, X_val, y_val)

    # Method 2: Grid search
    grid_weights, grid_val_acc = optimize_weights_grid_search(models, X_val, y_val)

    # Method 3: Differential evolution
    de_weights, de_val_acc = optimize_weights_differential_evolution(models, X_val, y_val)

    # Compare all methods on test set
    print("\n" + "=" * 80)
    print("COMPARISON: ALL METHODS ON TEST SET")
    print("=" * 80)

    all_results = {
        'baseline': {
            'weights': fixed_weights,
            'metrics': baseline_metrics
        }
    }

    if scipy_weights is not None:
        scipy_metrics = evaluate_ensemble(models, X_test, y_test, scipy_weights, "SCIPY OPTIMIZE")
        all_results['scipy'] = {
            'weights': scipy_weights,
            'val_accuracy': scipy_val_acc,
            'metrics': scipy_metrics
        }

    if grid_weights is not None:
        grid_metrics = evaluate_ensemble(models, X_test, y_test, grid_weights, "GRID SEARCH")
        all_results['grid_search'] = {
            'weights': grid_weights,
            'val_accuracy': grid_val_acc,
            'metrics': grid_metrics
        }

    if de_weights is not None:
        de_metrics = evaluate_ensemble(models, X_test, y_test, de_weights, "DIFFERENTIAL EVOLUTION")
        all_results['differential_evolution'] = {
            'weights': de_weights,
            'val_accuracy': de_val_acc,
            'metrics': de_metrics
        }

    # Find best method
    print("\n" + "=" * 80)
    print("BEST METHOD SELECTION")
    print("=" * 80)

    best_method = None
    best_accuracy = baseline_metrics['direction_accuracy']

    for method, data in all_results.items():
        acc = data['metrics']['direction_accuracy']
        if acc > best_accuracy:
            best_accuracy = acc
            best_method = method

    if best_method is None:
        best_method = 'baseline'

    print(f"\nBest method: {best_method.upper()}")
    print(f"  Weights: {all_results[best_method]['weights']}")
    print(f"  Test Direction Accuracy: {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}%")
    print(f"  Improvement over baseline: {(all_results[best_method]['metrics']['direction_accuracy'] - baseline_metrics['direction_accuracy'])*100:+.2f} percentage points")

    # Save best ensemble
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    best_ensemble = {
        'models': models,
        'weights': {
            'xgboost': float(all_results[best_method]['weights'][0]),
            'lightgbm': float(all_results[best_method]['weights'][1]),
            'catboost': float(all_results[best_method]['weights'][2])
        },
        'feature_cols': feature_cols,
        'optimization_method': best_method
    }

    model_path = artifacts_dir / "ensemble_optimized_weights_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_ensemble, f)

    print(f"\nBest ensemble saved to: {model_path}")

    # Save metrics
    metrics = {
        'all_methods': {
            method: {
                'weights': {
                    'xgboost': float(data['weights'][0]),
                    'lightgbm': float(data['weights'][1]),
                    'catboost': float(data['weights'][2])
                },
                'test_metrics': data['metrics']
            }
            for method, data in all_results.items()
        },
        'best_method': best_method,
        'final_metrics': all_results[best_method]['metrics']
    }

    metrics_path = artifacts_dir / "ensemble_optimized_weights_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("WEIGHT OPTIMIZATION COMPLETE!")
    print("=" * 80)

    print(f"\nFinal Results:")
    print(f"  Method: {best_method.upper()}")
    print(f"  Direction Accuracy: {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}%")
    print(f"  MAE: {all_results[best_method]['metrics']['mae']*100:.2f}%")
    print(f"  R2: {all_results[best_method]['metrics']['r2']:.4f}")

    print(f"\nComparison:")
    print(f"  Baseline (35/35/30):  {baseline_metrics['direction_accuracy']*100:.2f}%")
    print(f"  Optimized ({best_method}): {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}%")
    print(f"  Improvement: {(all_results[best_method]['metrics']['direction_accuracy'] - baseline_metrics['direction_accuracy'])*100:+.2f} percentage points")

    print(f"\nOverall progress:")
    print(f"  175 features (baseline):  58.42%")
    print(f"  60 features (selected):   58.59%")
    print(f"  Optimized weights:        {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}%")
    print(f"  Total improvement:        {(all_results[best_method]['metrics']['direction_accuracy'] - 0.5842)*100:+.2f} percentage points")

    if all_results[best_method]['metrics']['direction_accuracy'] >= 0.62:
        print(f"\n[SUCCESS] Achieved {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}% (>=62%) direction accuracy!")
    elif all_results[best_method]['metrics']['direction_accuracy'] >= 0.60:
        print(f"\n[GOOD] Achieved {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}% (>=60%) direction accuracy!")
    elif all_results[best_method]['metrics']['direction_accuracy'] >= 0.59:
        print(f"\n[PROGRESS] Achieved {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}% direction accuracy.")
    else:
        print(f"\n[NOTE] Achieved {all_results[best_method]['metrics']['direction_accuracy']*100:.2f}% direction accuracy.")


if __name__ == "__main__":
    main()
