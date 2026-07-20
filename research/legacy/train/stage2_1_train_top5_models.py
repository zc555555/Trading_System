"""
Stage 2.1: Train Dedicated Models for Top 5 Stocks

For each of the top 5 performing stocks, train a specialized model:
- Use only that stock's data
- Optimize features specific to that stock
- Tune hyperparameters for that stock's characteristics

Top 5 (from Stage 1.3):
1. NFLX: 62.72%
2. WMT: 62.47%
3. GS: 61.98%
4. HD: 61.98%
5. META: 61.48%

Expected: 66-68% -> 69-70%+
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def direction_accuracy(y_true, y_pred):
    return ((y_true > 0) == (y_pred > 0)).mean()


def train_stock_specific_model(stock_data, symbol, feature_cols):
    """
    Train a model specifically for one stock

    Returns:
    - models: dict of trained models
    - metrics: performance metrics
    """
    print(f"\n{'-' * 80}")
    print(f"Training model for {symbol}")
    print(f"{'-' * 80}")

    # Split data
    split_idx = int(len(stock_data) * 0.8)
    train_data = stock_data.iloc[:split_idx].copy()
    test_data = stock_data.iloc[split_idx:].copy()

    print(f"Train: {len(train_data)} samples")
    print(f"Test:  {len(test_data)} samples")

    # Prepare features
    X_train = train_data[feature_cols].values
    y_train = train_data['future_return'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=feature_cols).ffill().fillna(0).values

    # Sample weights (exponential)
    train_dates = pd.to_datetime(train_data['date'])
    days_from_start = (train_dates - train_dates.min()).dt.days
    max_days = days_from_start.max()
    alpha = np.log(3)
    sample_weights = np.exp(alpha * (days_from_start / max_days)).values

    models = {}
    predictions = {}

    # XGBoost
    print(f"\n  Training XGBoost...")
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
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = direction_accuracy(y_test, xgb_pred)
    print(f"    Test accuracy: {xgb_acc*100:.2f}%")
    models['xgboost'] = xgb_model
    predictions['xgboost'] = xgb_pred

    # LightGBM
    print(f"  Training LightGBM...")
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
    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = direction_accuracy(y_test, lgb_pred)
    print(f"    Test accuracy: {lgb_acc*100:.2f}%")
    models['lightgbm'] = lgb_model
    predictions['lightgbm'] = lgb_pred

    # CatBoost
    if HAS_CATBOOST:
        print(f"  Training CatBoost...")
        cat_model = cb.CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train, sample_weight=sample_weights)
        cat_pred = cat_model.predict(X_test)
        cat_acc = direction_accuracy(y_test, cat_pred)
        print(f"    Test accuracy: {cat_acc*100:.2f}%")
        models['catboost'] = cat_model
        predictions['catboost'] = cat_pred

    # Ensemble (simple average for now)
    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix.mean(axis=1)
    ensemble_acc = direction_accuracy(y_test, ensemble_pred)

    print(f"\n  Ensemble accuracy: {ensemble_acc*100:.2f}%")

    # Feature importance (from XGBoost)
    feature_importance = xgb_model.feature_importances_
    top_features = sorted(
        zip(feature_cols, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\n  Top 10 features for {symbol}:")
    for i, (feat, imp) in enumerate(top_features, 1):
        print(f"    {i:2d}. {feat:40s} {imp:.6f}")

    metrics = {
        'symbol': symbol,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'individual_accuracies': {
            m: float(direction_accuracy(y_test, predictions[m]))
            for m in predictions.keys()
        },
        'ensemble_accuracy': float(ensemble_acc),
        'top_10_features': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in top_features
        ]
    }

    return models, metrics


def main():
    print("\n" + "=" * 80)
    print("Stage 2.1: Train Dedicated Models for Top 5 Stocks")
    print("=" * 80)

    # Load Stage 1.3 results to get Top 5
    artifacts_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/artifacts")
    stage1_results_path = artifacts_dir / "stage1_3_ensemble_signals_results.json"

    with open(stage1_results_path, 'r') as f:
        stage1_results = json.load(f)

    top5_stocks = stage1_results['core_stocks']
    print(f"\nTop 5 stocks (from Stage 1.3): {top5_stocks}")

    # Load data
    data_dir = Path("c:/Users/13785/OneDrive/Desktop/stock_predict/research/data")
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # Load general model for feature columns
    general_model_path = artifacts_dir / "ensemble_time_windows.pkl"
    with open(general_model_path, 'rb') as f:
        ensemble_dict = pickle.load(f)
    feature_cols = ensemble_dict['feature_cols']

    print(f"\nTotal stocks in dataset: {df['symbol'].nunique()}")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {len(feature_cols)}")

    # Train models for each Top 5 stock
    all_stock_models = {}
    all_metrics = {}

    for symbol in top5_stocks:
        stock_data = df[df['symbol'] == symbol].copy()

        if len(stock_data) < 100:
            print(f"\n[SKIP] {symbol}: Not enough data ({len(stock_data)} samples)")
            continue

        models, metrics = train_stock_specific_model(stock_data, symbol, feature_cols)

        all_stock_models[symbol] = models
        all_metrics[symbol] = metrics

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(f"\n{'Stock':<8} {'Samples':<10} {'Ensemble Acc':<15} {'Best Model':<12}")
    print("-" * 50)

    for symbol in top5_stocks:
        if symbol in all_metrics:
            metrics = all_metrics[symbol]
            best_model = max(
                metrics['individual_accuracies'].items(),
                key=lambda x: x[1]
            )
            print(f"{symbol:<8} {metrics['test_samples']:<10} "
                  f"{metrics['ensemble_accuracy']*100:>6.2f}%        "
                  f"{best_model[0]:<12} ({best_model[1]*100:.2f}%)")

    # Save models
    print("\n" + "-" * 80)
    print("Saving models...")
    print("-" * 80)

    for symbol, models in all_stock_models.items():
        # Save each stock's ensemble
        stock_ensemble = {
            'symbol': symbol,
            'models': models,
            'feature_cols': feature_cols,
            'model_type': 'stock_specific'
        }

        model_path = artifacts_dir / f"stock_model_{symbol}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(stock_ensemble, f)

        print(f"  Saved: {model_path}")

    # Save metrics
    metrics_path = artifacts_dir / "stage2_1_top5_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'top5_stocks': top5_stocks,
            'stock_metrics': all_metrics
        }, f, indent=2)

    print(f"\nMetrics saved: {metrics_path}")

    # Calculate average improvement
    avg_acc = np.mean([m['ensemble_accuracy'] for m in all_metrics.values()])
    print(f"\n{'=' * 80}")
    print(f"Average accuracy across Top 5 stocks: {avg_acc*100:.2f}%")
    print(f"{'=' * 80}")

    print(f"\n[COMPLETE] Stage 2.1 finished!")
    print(f"Top 5 stock-specific models trained and saved.")
    print(f"\nNext: Stage 3.1 - Market Regime Classification")


if __name__ == "__main__":
    main()
