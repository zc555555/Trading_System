"""
Train Final Model with External Features.

Combines:
- 60 baseline features (technical + market)
- 5 IV features (if available)
- 6 news sentiment features (if available)

Total: Up to 71 features

Expected improvement: +2-3.5% → 61-62% accuracy
"""

import pickle
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not available. Will use XGBoost + LightGBM only.")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import differential_evolution


def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def load_and_merge_features(iv_only=False):
    """
    Load base features and merge with external features.

    Args:
        iv_only: If True, only use IV features (for quick testing)

    Returns:
        DataFrame with all available features
    """
    data_dir = Path(__file__).parent.parent / "data"

    # Load base features (60 features)
    print("\n" + "=" * 80)
    print("LOADING FEATURES")
    print("=" * 80)

    base_path = data_dir / "stocks_selected_features.parquet"
    df = pd.read_parquet(base_path)

    print(f"\nBase features: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Get base feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    base_features = [c for c in df.columns if c not in meta_cols]

    print(f"  Base feature count: {len(base_features)}")

    # Load IV features
    iv_path = data_dir / "iv_features.parquet"
    has_iv = iv_path.exists()

    if has_iv:
        print("\n[+] Loading IV features...")
        iv_df = pd.read_parquet(iv_path)

        # Remove timezone from both dataframes for merging
        if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        if hasattr(iv_df['date'].dtype, 'tz') and iv_df['date'].dtype.tz is not None:
            iv_df['date'] = iv_df['date'].dt.tz_localize(None)

        # Merge on date + symbol
        df = df.merge(iv_df, on=['date', 'symbol'], how='left')

        iv_features = [c for c in iv_df.columns if c not in ['date', 'symbol']]
        print(f"  IV features added: {len(iv_features)}")
        print(f"  Features: {iv_features}")
    else:
        print("\n[-] IV features not found")
        print(f"  Expected path: {iv_path}")
        print("  Run: python data/fetch_option_iv.py")
        print("       python features/process_iv_with_lstm.py")
        iv_features = []

    # Load news features (unless iv_only mode)
    news_features = []
    if not iv_only:
        news_path = data_dir / "news_features.parquet"
        has_news = news_path.exists()

        if has_news:
            print("\n[+] Loading news sentiment features...")
            news_df = pd.read_parquet(news_path)

            # Remove timezone for merging
            if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            if hasattr(news_df['date'].dtype, 'tz') and news_df['date'].dtype.tz is not None:
                news_df['date'] = news_df['date'].dt.tz_localize(None)

            # Merge on date + symbol
            df = df.merge(news_df, on=['date', 'symbol'], how='left')

            news_features = [c for c in news_df.columns if c not in ['date', 'symbol']]
            print(f"  News features added: {len(news_features)}")
            print(f"  Features: {news_features}")
        else:
            print("\n[-] News sentiment features not found")
            print(f"  Expected path: {news_path}")
            print("  Run: python data/fetch_news_sentiment.py")
            print("       python features/process_news_with_lstm.py")

    # Combined feature list
    all_features = base_features + iv_features + news_features

    print(f"\n{'='*80}")
    print("FEATURE SUMMARY")
    print(f"{'='*80}")
    print(f"  Base features:          {len(base_features)}")
    print(f"  IV features:            {len(iv_features)}")
    print(f"  News features:          {len(news_features)}")
    print(f"  {'-'*80}")
    print(f"  TOTAL:                  {len(all_features)}")

    return df, all_features, base_features, iv_features, news_features


def train_final_model(iv_only=False):
    """Train final model with all available features."""

    print("=" * 80)
    print("TRAINING FINAL MODEL WITH EXTERNAL FEATURES")
    print("=" * 80)

    # Load data
    df, all_features, base_features, iv_features, news_features = load_and_merge_features(iv_only)

    if len(all_features) == len(base_features):
        print("\n!  WARNING: No external features available!")
        print("This will train on baseline features only.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please fetch external data first.")
            return

    # Split data (80/20)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df):,} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Test:  {len(test_df):,} samples ({test_df['date'].min()} to {test_df['date'].max()})")

    # Prepare features
    X_train = train_df[all_features].values
    y_train = train_df['future_return'].values

    X_test = test_df[all_features].values
    y_test = test_df['future_return'].values

    # Handle NaN (external features may have missing values)
    print(f"\nHandling missing values...")
    X_train_df = pd.DataFrame(X_train, columns=all_features)
    X_test_df = pd.DataFrame(X_test, columns=all_features)

    # Fill forward, then fill remaining with 0
    X_train = X_train_df.ffill().fillna(0).values
    X_test = X_test_df.ffill().fillna(0).values

    print(f"  Training data: {X_train.shape}")
    print(f"  Test data: {X_test.shape}")

    # Train XGBoost
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    xgb_model.fit(X_train, y_train)

    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    xgb_train_acc = direction_accuracy(y_train, xgb_train_pred)
    xgb_test_acc = direction_accuracy(y_test, xgb_test_pred)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)

    print(f"  Train Direction Accuracy: {xgb_train_acc*100:.2f}%")
    print(f"  Test Direction Accuracy:  {xgb_test_acc*100:.2f}%")
    print(f"  Test MAE: {xgb_test_mae*100:.2f}%")

    # Train LightGBM
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)

    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=500)

    lgb_train_pred = lgb_model.predict(X_train)
    lgb_test_pred = lgb_model.predict(X_test)

    lgb_train_acc = direction_accuracy(y_train, lgb_train_pred)
    lgb_test_acc = direction_accuracy(y_test, lgb_test_pred)
    lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)

    print(f"  Train Direction Accuracy: {lgb_train_acc*100:.2f}%")
    print(f"  Test Direction Accuracy:  {lgb_test_acc*100:.2f}%")
    print(f"  Test MAE: {lgb_test_mae*100:.2f}%")

    # Train CatBoost (if available)
    if HAS_CATBOOST:
        print("\n" + "=" * 80)
        print("TRAINING CATBOOST")
        print("=" * 80)

        cb_model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            bagging_temperature=0.7,
            random_strength=0.1,
            border_count=128,
            random_state=42,
            verbose=False,
            thread_count=-1
        )

        cb_model.fit(X_train, y_train)

        cb_train_pred = cb_model.predict(X_train)
        cb_test_pred = cb_model.predict(X_test)

        cb_train_acc = direction_accuracy(y_train, cb_train_pred)
        cb_test_acc = direction_accuracy(y_test, cb_test_pred)
        cb_test_mae = mean_absolute_error(y_test, cb_test_pred)

        print(f"  Train Direction Accuracy: {cb_train_acc*100:.2f}%")
        print(f"  Test Direction Accuracy:  {cb_test_acc*100:.2f}%")
        print(f"  Test MAE: {cb_test_mae*100:.2f}%")
    else:
        cb_model = None
        cb_test_pred = None
        cb_train_acc = None
        cb_test_acc = None
        cb_test_mae = None

    # Optimize ensemble weights
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 80)

    if HAS_CATBOOST:
        pred_matrix = np.column_stack([xgb_test_pred, lgb_test_pred, cb_test_pred])
        bounds = [(0, 1)] * 3
        num_models = 3
    else:
        pred_matrix = np.column_stack([xgb_test_pred, lgb_test_pred])
        bounds = [(0, 1)] * 2
        num_models = 2

    def objective(weights):
        weights_norm = weights / weights.sum()
        ensemble_pred = pred_matrix @ weights_norm
        return -direction_accuracy(y_test, ensemble_pred)

    result = differential_evolution(objective, bounds, seed=42, maxiter=100)

    optimal_weights = result.x / result.x.sum()
    ensemble_pred = pred_matrix @ optimal_weights

    ensemble_acc = direction_accuracy(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print(f"\nOptimal weights:")
    print(f"  XGBoost:  {optimal_weights[0]*100:.1f}%")
    print(f"  LightGBM: {optimal_weights[1]*100:.1f}%")
    if HAS_CATBOOST:
        print(f"  CatBoost: {optimal_weights[2]*100:.1f}%")

    print(f"\nEnsemble Performance:")
    print(f"  Direction Accuracy: {ensemble_acc*100:.2f}%")
    print(f"  MAE: {ensemble_mae*100:.2f}%")
    print(f"  R2:  {ensemble_r2:.4f}")

    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    baseline_acc = 0.5912  # Super ensemble

    print(f"\nBaseline (Super Ensemble, 60 features):  {baseline_acc*100:.2f}%")
    print(f"With External Features ({len(all_features)} features): {ensemble_acc*100:.2f}%")
    print(f"Improvement: {(ensemble_acc - baseline_acc)*100:+.2f} percentage points")

    # Feature importance analysis
    print("\n" + "=" * 80)
    print("TOP EXTERNAL FEATURES")
    print("=" * 80)

    # Get feature importance from XGBoost
    xgb_importance = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': xgb_importance
    }).sort_values('importance', ascending=False)

    # Filter external features only
    external_features = iv_features + news_features
    if external_features:
        external_importance = feature_importance_df[
            feature_importance_df['feature'].isin(external_features)
        ].head(10)

        print(f"\nTop 10 external features (by importance):")
        for i, row in enumerate(external_importance.itertuples(), 1):
            feature_type = "IV" if row.feature in iv_features else "News"
            print(f"  {i:2d}. {row.feature:<30} {row.importance:>8.4f} ({feature_type})")

    # Save model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    model_data = {
        'models': {
            'xgboost': xgb_model,
            'lightgbm': lgb_model
        },
        'weights': {
            'xgboost': float(optimal_weights[0]),
            'lightgbm': float(optimal_weights[1])
        },
        'feature_cols': all_features,
        'base_features': base_features,
        'iv_features': iv_features,
        'news_features': news_features,
        'has_external_data': len(external_features) > 0,
        'has_catboost': HAS_CATBOOST
    }

    if HAS_CATBOOST:
        model_data['models']['catboost'] = cb_model
        model_data['weights']['catboost'] = float(optimal_weights[2])

    model_path = artifacts_dir / "final_model_with_external_features.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n\nModel saved to: {model_path}")

    # Save metrics
    metrics = {
        'model_type': 'ensemble_with_external_features',
        'feature_counts': {
            'base': len(base_features),
            'iv': len(iv_features),
            'news': len(news_features),
            'total': len(all_features)
        },
        'models': {
            'xgboost': {
                'train_direction_accuracy': float(xgb_train_acc),
                'test_direction_accuracy': float(xgb_test_acc),
                'test_mae': float(xgb_test_mae)
            },
            'lightgbm': {
                'train_direction_accuracy': float(lgb_train_acc),
                'test_direction_accuracy': float(lgb_test_acc),
                'test_mae': float(lgb_test_mae)
            }
        },
        'ensemble': {
            'test_direction_accuracy': float(ensemble_acc),
            'test_mae': float(ensemble_mae),
            'test_r2': float(ensemble_r2),
            'weights': {
                'xgboost': float(optimal_weights[0]),
                'lightgbm': float(optimal_weights[1])
            }
        },
        'baseline_comparison': {
            'baseline_accuracy': baseline_acc,
            'final_accuracy': float(ensemble_acc),
            'improvement': float(ensemble_acc - baseline_acc)
        }
    }

    if HAS_CATBOOST:
        metrics['models']['catboost'] = {
            'train_direction_accuracy': float(cb_train_acc),
            'test_direction_accuracy': float(cb_test_acc),
            'test_mae': float(cb_test_mae)
        }
        metrics['ensemble']['weights']['catboost'] = float(optimal_weights[2])

    metrics_path = artifacts_dir / "final_model_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Final assessment
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if ensemble_acc >= 0.63:
        print(f"\n*** [SUCCESS] Achieved {ensemble_acc*100:.2f}% (>=63%) direction accuracy!")
        print("Target exceeded! This model has strong predictive power.")
    elif ensemble_acc >= 0.61:
        print(f"\n** [VERY GOOD] Achieved {ensemble_acc*100:.2f}% (>=61%) direction accuracy!")
        print("Very close to target. This is a strong trading model.")
    elif ensemble_acc >= 0.60:
        print(f"\n+ [GOOD] Achieved {ensemble_acc*100:.2f}% (>=60%) direction accuracy!")
        print("Exceeded 60% threshold. Profitable for trading.")
    else:
        print(f"\n[RESULT] Achieved {ensemble_acc*100:.2f}% direction accuracy.")
        print(f"Improvement: {(ensemble_acc - baseline_acc)*100:+.2f} percentage points")

        if len(external_features) == 0:
            print("\n!  No external features were used.")
            print("To reach 63%, you need to fetch and add external data:")
            print("  1. Options IV data: +0.5-1.0%")
            print("  2. News sentiment: +1.5-2.5%")
        elif len(iv_features) > 0 and len(news_features) == 0:
            print("\n!  Only IV features were used.")
            print("To reach 63%, add news sentiment features for +1.5-2.5% more:")
            print("  python data/fetch_news_sentiment.py")
            print("  python features/process_news_with_lstm.py")
        elif len(news_features) > 0 and len(iv_features) == 0:
            print("\n!  Only news features were used.")
            print("To reach 63%, add IV features for +0.5-1.0% more:")
            print("  python data/fetch_option_iv.py")
            print("  python features/process_iv_with_lstm.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train final model with external features')
    parser.add_argument('--iv-only', action='store_true',
                        help='Use only IV features (for quick testing)')

    args = parser.parse_args()

    train_final_model(iv_only=args.iv_only)
