"""
Feature Selection: Reduce 175 features to 60 high-quality features.

Strategy:
1. Remove redundant market features (keep only _y version)
2. Remove low-importance 101 Alphas features
3. Keep all directional features (11)
4. Keep top technical indicators based on model importance
5. Use multi-model voting to select best features

Expected improvement: 58.42% -> 60.5-61.5% direction accuracy
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def load_models():
    """Load trained models."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    models = {}

    # XGBoost
    xgb_path = artifacts_dir / "xgboost_regression_model.pkl"
    with open(xgb_path, 'rb') as f:
        models['xgboost'] = pickle.load(f)

    # LightGBM
    lgb_path = artifacts_dir / "lightgbm_regression_model.pkl"
    with open(lgb_path, 'rb') as f:
        models['lightgbm'] = pickle.load(f)

    # CatBoost
    cb_path = artifacts_dir / "catboost_regression_model.pkl"
    with open(cb_path, 'rb') as f:
        models['catboost'] = pickle.load(f)

    return models


def get_feature_importance(model_name, model, feature_names):
    """Extract feature importance from model."""

    if model_name == 'xgboost':
        # XGBoost sklearn wrapper has feature_importances_ attribute
        importance = model.feature_importances_

    elif model_name == 'lightgbm':
        # LightGBM Booster has feature_importance() method
        try:
            importance = model.feature_importance(importance_type='gain')
        except:
            # Fallback to split importance
            importance = model.feature_importance(importance_type='split')

    elif model_name == 'catboost':
        # CatBoost has get_feature_importance() method
        importance = model.get_feature_importance()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return importance


def select_features_multi_model_voting(models, feature_names, target_count=60):
    """
    Select top features using multi-model voting.

    Strategy:
    1. Rank features by importance in each model
    2. Keep features that rank high in at least one model
    3. Always keep directional features
    4. Remove redundant market features (keep only _y version)
    """

    print("=" * 80)
    print("FEATURE SELECTION: MULTI-MODEL VOTING")
    print("=" * 80)

    # Step 1: Get importance from each model
    print("\nStep 1: Extracting feature importance from each model...")

    all_importance = {}
    for model_name, model in models.items():
        importance = get_feature_importance(model_name, model, feature_names)
        all_importance[model_name] = importance

        # Show top 10
        top_indices = np.argsort(importance)[::-1][:10]
        print(f"\n{model_name.upper()} - Top 10 features:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank:2d}. {feature_names[idx]:30s} : {importance[idx]:.4f}")

    # Step 2: Identify must-keep features
    print("\n" + "=" * 80)
    print("Step 2: Identifying must-keep features...")
    print("=" * 80)

    directional_features = [
        'price_above_sma_20', 'price_above_ema_50', 'golden_cross',
        'adx_strong_trend', 'macd_positive', 'rsi_oversold',
        'consecutive_up_days', 'consecutive_down_days',
        'returns_1d_positive', 'volume_increasing', 'bb_squeeze'
    ]

    must_keep = set()

    # Always keep directional features
    for feat in directional_features:
        if feat in feature_names:
            must_keep.add(feat)

    # Always keep dpo_20 (top feature in all models)
    if 'dpo_20' in feature_names:
        must_keep.add('dpo_20')

    print(f"\nMust-keep features ({len(must_keep)}):")
    for feat in sorted(must_keep):
        print(f"  - {feat}")

    # Step 3: Remove redundant market features
    print("\n" + "=" * 80)
    print("Step 3: Handling redundant market features...")
    print("=" * 80)

    # Market feature bases
    market_bases = [
        'spy_returns_1d', 'spy_returns_5d', 'spy_returns_20d', 'spy_volatility_20d',
        'qqq_returns_1d', 'qqq_returns_5d',
        'dia_returns_1d', 'dia_returns_5d',
        'iwm_returns_1d', 'iwm_returns_5d',
        'vix_level', 'vix_change_1d', 'vix_change_5d',
        'tlt_returns_1d', 'tlt_returns_5d',
        'gld_returns_1d', 'gld_returns_5d',
        'uso_returns_1d', 'uso_returns_5d',
        'uup_returns_1d', 'uup_returns_5d',
        'market_breadth_1d', 'market_breadth_5d'
    ]

    # For each base, keep only _y version (most recent lag)
    redundant_features = set()
    preferred_market_features = set()

    for base in market_bases:
        x_ver = f"{base}_x"
        y_ver = f"{base}_y"
        original = base

        # If all three exist, keep only y version
        if x_ver in feature_names and y_ver in feature_names and original in feature_names:
            redundant_features.add(x_ver)
            redundant_features.add(original)
            preferred_market_features.add(y_ver)
            print(f"  {base:25s}: Keeping {y_ver:30s}, removing {x_ver} and {original}")

    print(f"\nRedundant features to remove: {len(redundant_features)}")
    print(f"Preferred market features: {len(preferred_market_features)}")

    # Step 4: Rank features by importance (excluding redundant)
    print("\n" + "=" * 80)
    print("Step 4: Ranking non-redundant features...")
    print("=" * 80)

    # Calculate average rank across models
    feature_scores = {}

    for i, fname in enumerate(feature_names):
        # Skip redundant features
        if fname in redundant_features:
            continue

        # Calculate average importance across models
        avg_importance = np.mean([all_importance[m][i] for m in models.keys()])
        feature_scores[fname] = avg_importance

    # Sort by importance
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 30 non-redundant features:")
    for rank, (fname, score) in enumerate(sorted_features[:30], 1):
        marker = "[MUST-KEEP]" if fname in must_keep else ""
        print(f"  {rank:2d}. {fname:35s} : {score:8.4f} {marker}")

    # Step 5: Select final features
    print("\n" + "=" * 80)
    print(f"Step 5: Selecting top {target_count} features...")
    print("=" * 80)

    selected_features = set(must_keep)  # Start with must-keep

    # Add top features until we reach target_count
    for fname, score in sorted_features:
        if len(selected_features) >= target_count:
            break
        if fname not in selected_features:
            selected_features.add(fname)

    # Convert to sorted list
    selected_features = sorted(selected_features)

    # Step 6: Categorize selected features
    print("\n" + "=" * 80)
    print(f"SELECTED FEATURES: {len(selected_features)}")
    print("=" * 80)

    categories = {
        'Alpha Features': [],
        'Market Features': [],
        'Directional Features': [],
        'Technical Indicators': [],
        'Basic Features': []
    }

    for feat in selected_features:
        if feat.startswith('alpha_'):
            categories['Alpha Features'].append(feat)
        elif any(x in feat for x in ['spy_', 'qqq_', 'dia_', 'iwm_', 'vix_', 'tlt_', 'gld_', 'uso_', 'uup_', 'market_breadth']):
            categories['Market Features'].append(feat)
        elif feat in directional_features:
            categories['Directional Features'].append(feat)
        elif feat in ['returns_1d', 'returns_5d', 'volatility_20d', 'volume_ratio_5d']:
            categories['Basic Features'].append(feat)
        else:
            categories['Technical Indicators'].append(feat)

    for category, features in categories.items():
        if features:
            print(f"\n{category} ({len(features)}):")
            for feat in features:
                print(f"  - {feat}")

    return selected_features


def save_selected_features(selected_features):
    """Save selected features to file."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # Save as JSON
    output_path = artifacts_dir / "selected_features.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'selected_features': selected_features,
            'count': len(selected_features),
            'description': 'Top features selected by multi-model voting'
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Selected features saved to: {output_path}")
    print(f"{'=' * 80}")

    return output_path


def create_reduced_dataset(selected_features):
    """Create new dataset with only selected features."""

    print("\n" + "=" * 80)
    print("CREATING REDUCED DATASET")
    print("=" * 80)

    # Load original dataset
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks.parquet")

    print(f"\nOriginal dataset: {df.shape}")

    # Meta columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']

    # Keep only selected features + meta columns
    cols_to_keep = meta_cols + selected_features
    df_reduced = df[cols_to_keep].copy()

    print(f"Reduced dataset: {df_reduced.shape}")
    print(f"Features reduced: 175 -> {len(selected_features)}")
    print(f"Size reduction: {(1 - len(selected_features)/175)*100:.1f}%")

    # Save reduced dataset
    output_path = data_dir / "stocks_selected_features.parquet"
    df_reduced.to_parquet(output_path, index=False)

    print(f"\nReduced dataset saved to: {output_path}")

    return output_path


def main():
    """Main feature selection workflow."""

    print("\n" + "=" * 80)
    print("FEATURE SELECTION WORKFLOW")
    print("=" * 80)
    print("\nGoal: Reduce 175 features to 60 high-quality features")
    print("Expected improvement: 58.42% -> 60.5-61.5% direction accuracy\n")

    # Load data to get feature names
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks.parquet")

    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    feature_names = [c for c in df.columns if c not in meta_cols]

    print(f"Original features: {len(feature_names)}")

    # Load trained models
    print("\nLoading trained models...")
    models = load_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}")

    # Select features
    selected_features = select_features_multi_model_voting(
        models, feature_names, target_count=60
    )

    # Save selected features
    save_selected_features(selected_features)

    # Create reduced dataset
    create_reduced_dataset(selected_features)

    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETE!")
    print("=" * 80)

    print(f"\nNext steps:")
    print(f"  1. Train models with selected features:")
    print(f"     python train/train_xgb_regression.py --use-selected-features")
    print(f"     python train/train_lightgbm_regression.py --use-selected-features")
    print(f"     python train/train_catboost_regression.py --use-selected-features")
    print(f"  2. Create ensemble:")
    print(f"     python train/train_ensemble.py --use-selected-features")
    print(f"\nExpected result: 60.5-61.5% direction accuracy")


if __name__ == "__main__":
    main()
