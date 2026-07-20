"""
Train with Feature Interactions.

Create meaningful feature combinations:
- Price momentum * Volume
- Market features * Individual features
- Technical indicator combinations

Expected improvement: +1-2% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy."""
    return ((y_true > 0) == (y_pred > 0)).mean()


def create_interaction_features(df, feature_cols):
    """
    Create meaningful feature interactions.

    Strategy:
    1. Market * Individual: spy_returns * dpo_20
    2. Momentum * Volume: returns * volume_ratio
    3. Volatility * Trend: volatility * rsi
    4. Cross-market: spy * vix
    """

    print("\nCreating feature interactions...")

    new_features = {}

    # Get specific features
    dpo_20 = df['dpo_20'].values if 'dpo_20' in feature_cols else None

    # Market features
    spy_returns_1d = df['spy_returns_1d_y'].values if 'spy_returns_1d_y' in feature_cols else None
    vix_level = df['vix_level_y'].values if 'vix_level_y' in feature_cols else None

    # Volume features
    volume_ratio_5d = df['volume_ratio_5d'].values if 'volume_ratio_5d' in feature_cols else None

    # Returns features
    returns_5d = df['returns_5d'].values if 'returns_5d' in feature_cols else None

    # Volatility features
    volatility_20d = df['volatility_20d'].values if 'volatility_20d' in feature_cols else None

    # RSI features
    rsi_7 = df['rsi_7'].values if 'rsi_7' in feature_cols else None

    # 1. Market * Individual interactions
    if spy_returns_1d is not None and dpo_20 is not None:
        new_features['spy_x_dpo'] = spy_returns_1d * dpo_20

    if vix_level is not None and dpo_20 is not None:
        new_features['vix_x_dpo'] = vix_level * dpo_20

    # 2. Momentum * Volume interactions
    if returns_5d is not None and volume_ratio_5d is not None:
        new_features['momentum_x_volume'] = returns_5d * volume_ratio_5d

    # 3. Volatility * Trend interactions
    if volatility_20d is not None and rsi_7 is not None:
        new_features['volatility_x_rsi'] = volatility_20d * rsi_7

    # 4. Market regime interactions
    if spy_returns_1d is not None and vix_level is not None:
        new_features['market_regime'] = spy_returns_1d * vix_level

    # 5. Ratio features
    if vix_level is not None and volatility_20d is not None:
        new_features['vix_volatility_ratio'] = np.divide(
            vix_level,
            volatility_20d + 1e-6,
            out=np.zeros_like(vix_level),
            where=(volatility_20d + 1e-6) != 0
        )

    # 6. Squared features (capture non-linearity)
    if dpo_20 is not None:
        new_features['dpo_20_squared'] = dpo_20 ** 2

    if returns_5d is not None:
        new_features['returns_5d_squared'] = returns_5d ** 2

    # 7. Market breadth interactions
    if 'market_breadth_1d_y' in df.columns and 'spy_returns_1d_y' in df.columns:
        new_features['breadth_x_spy'] = df['market_breadth_1d_y'].values * df['spy_returns_1d_y'].values

    # 8. Directional features * momentum
    if 'consecutive_up_days' in df.columns and returns_5d is not None:
        new_features['momentum_trend'] = df['consecutive_up_days'].values * returns_5d

    # 9. Cross-ETF interactions
    if 'spy_returns_1d_y' in df.columns and 'qqq_returns_1d_y' in df.columns:
        new_features['spy_qqq_correlation'] = df['spy_returns_1d_y'].values * df['qqq_returns_1d_y'].values

    # 10. VIX regime features
    if vix_level is not None:
        new_features['high_vix_regime'] = (vix_level > 20).astype(float)
        new_features['extreme_vix_regime'] = (vix_level > 30).astype(float)

    print(f"Created {len(new_features)} interaction features")

    return new_features


def train_with_interactions():
    """Train models with feature interactions."""

    print("=" * 80)
    print("TRAINING WITH FEATURE INTERACTIONS")
    print("=" * 80)
    print("\nStrategy: Create meaningful feature combinations")
    print("Expected improvement: +1-2% direction accuracy")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "stocks_selected_features.parquet"
    df = pd.read_parquet(data_path)

    print(f"\nOriginal dataset: {df.shape}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    original_feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Original features: {len(original_feature_cols)}")

    # Split data first (80/20)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    print(f"\nTrain: {len(train_df):,} samples")
    print(f"Test:  {len(test_df):,} samples")

    # Create interactions for train set
    print("\nCreating interactions for training set...")
    train_interactions = create_interaction_features(train_df, original_feature_cols)

    # Add interaction features to train_df
    for feat_name, feat_values in train_interactions.items():
        train_df[feat_name] = feat_values

    # Create interactions for test set
    print("Creating interactions for test set...")
    test_interactions = create_interaction_features(test_df, original_feature_cols)

    # Add interaction features to test_df
    for feat_name, feat_values in test_interactions.items():
        test_df[feat_name] = feat_values

    # Combined feature columns
    all_feature_cols = original_feature_cols + list(train_interactions.keys())

    print(f"\nTotal features after interactions: {len(all_feature_cols)}")

    # Prepare data
    X_train = train_df[all_feature_cols].values
    y_train = train_df['future_return'].values

    X_test = test_df[all_feature_cols].values
    y_test = test_df['future_return'].values

    # Handle NaN
    X_train = pd.DataFrame(X_train, columns=all_feature_cols).ffill().fillna(0).values
    X_test = pd.DataFrame(X_test, columns=all_feature_cols).ffill().fillna(0).values

    print(f"\nFinal training data: {X_train.shape}")
    print(f"Final test data: {X_test.shape}")

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

    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_acc = direction_accuracy(y_test, xgb_test_pred)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)

    print(f"  Test Direction Accuracy: {xgb_test_acc*100:.2f}%")
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

    lgb_test_pred = lgb_model.predict(X_test)
    lgb_test_acc = direction_accuracy(y_test, lgb_test_pred)
    lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)

    print(f"  Test Direction Accuracy: {lgb_test_acc*100:.2f}%")
    print(f"  Test MAE: {lgb_test_mae*100:.2f}%")

    # Train CatBoost
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

    cb_test_pred = cb_model.predict(X_test)
    cb_test_acc = direction_accuracy(y_test, cb_test_pred)
    cb_test_mae = mean_absolute_error(y_test, cb_test_pred)

    print(f"  Test Direction Accuracy: {cb_test_acc*100:.2f}%")
    print(f"  Test MAE: {cb_test_mae*100:.2f}%")

    # Ensemble
    print("\n" + "=" * 80)
    print("ENSEMBLE EVALUATION")
    print("=" * 80)

    ensemble_pred = (xgb_test_pred + lgb_test_pred + cb_test_pred) / 3
    ensemble_acc = direction_accuracy(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print(f"\nEnsemble (Equal Weights):")
    print(f"  Direction Accuracy: {ensemble_acc*100:.2f}%")
    print(f"  MAE: {ensemble_mae*100:.2f}%")
    print(f"  R2:  {ensemble_r2:.4f}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    baseline_acc = 0.5859  # 60 features, no interactions

    print(f"\nBaseline (60 features):              {baseline_acc*100:.2f}%")
    print(f"With Interactions ({len(all_feature_cols)} features): {ensemble_acc*100:.2f}%")
    print(f"Improvement: {(ensemble_acc - baseline_acc)*100:+.2f} percentage points")

    # Save models
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    models = {
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cb_model,
        'weights': {'xgboost': 1/3, 'lightgbm': 1/3, 'catboost': 1/3},
        'feature_cols': all_feature_cols,
        'interaction_features': list(train_interactions.keys())
    }

    model_path = artifacts_dir / "ensemble_with_interactions_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)

    print(f"\nModels saved to: {model_path}")

    # Save metrics
    metrics = {
        'feature_interactions': True,
        'original_features': len(original_feature_cols),
        'interaction_features': len(train_interactions),
        'total_features': len(all_feature_cols),
        'models': {
            'xgboost': {
                'test_direction_accuracy': float(xgb_test_acc),
                'test_mae': float(xgb_test_mae)
            },
            'lightgbm': {
                'test_direction_accuracy': float(lgb_test_acc),
                'test_mae': float(lgb_test_mae)
            },
            'catboost': {
                'test_direction_accuracy': float(cb_test_acc),
                'test_mae': float(cb_test_mae)
            }
        },
        'ensemble': {
            'test_direction_accuracy': float(ensemble_acc),
            'test_mae': float(ensemble_mae),
            'test_r2': float(ensemble_r2)
        },
        'baseline_comparison': {
            'baseline_accuracy': baseline_acc,
            'interaction_accuracy': float(ensemble_acc),
            'improvement': float(ensemble_acc - baseline_acc)
        }
    }

    metrics_path = artifacts_dir / "feature_interactions_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("FEATURE INTERACTION TRAINING COMPLETE!")
    print("=" * 80)

    if ensemble_acc >= 0.63:
        print(f"\n[SUCCESS] Achieved {ensemble_acc*100:.2f}% (>=63%) direction accuracy!")
    elif ensemble_acc >= 0.60:
        print(f"\n[GOOD] Achieved {ensemble_acc*100:.2f}% (>=60%) direction accuracy!")
    else:
        print(f"\n[RESULT] Achieved {ensemble_acc*100:.2f}% direction accuracy.")


if __name__ == "__main__":
    train_with_interactions()
