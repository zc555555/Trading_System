"""
MULTI-FACTOR MODEL TRAINING

Train separate ensemble models for each factor category:
- Momentum factor model
- Trend factor model
- Volatility factor model
- Volume factor model
- Market factor model
- Alpha factor model

Each factor has its own XGBoost + LightGBM + CatBoost ensemble.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

# Import factor definitions
from factors.factor_definitions import (
    FACTOR_GROUPS,
    FACTOR_WEIGHTS,
    verify_feature_coverage
)


def train_ensemble_for_factor(X_train, y_train, X_val, y_val, factor_name):
    """
    Train an ensemble (XGB + LGB + CAT) for a specific factor.

    Returns:
        dict: {
            'models': {'xgb': model, 'lgb': model, 'cat': model},
            'weights': {'xgb': w, 'lgb': w, 'cat': w},
            'val_score': score
        }
    """
    print(f"\n{'='*80}")
    print(f"Training {factor_name.upper()} factor ensemble")
    print(f"{'='*80}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    models = {}

    # XGBoost
    print("\n[1/3] Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)
    xgb_score = np.corrcoef(xgb_pred, y_val)[0, 1]
    models['xgb'] = xgb
    print(f"[OK] XGBoost correlation: {xgb_score:.4f}")

    # LightGBM
    print("\n[2/3] Training LightGBM...")
    lgb = LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_val)
    lgb_score = np.corrcoef(lgb_pred, y_val)[0, 1]
    models['lgb'] = lgb
    print(f"[OK] LightGBM correlation: {lgb_score:.4f}")

    # CatBoost
    print("\n[3/3] Training CatBoost...")
    cat = CatBoostRegressor(
        iterations=100,
        depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
    cat.fit(X_train, y_train)
    cat_pred = cat.predict(X_val)
    cat_score = np.corrcoef(cat_pred, y_val)[0, 1]
    models['cat'] = cat
    print(f"[OK] CatBoost correlation: {cat_score:.4f}")

    # Calculate ensemble weights based on validation performance
    scores = {'xgb': xgb_score, 'lgb': lgb_score, 'cat': cat_score}
    total_score = sum(scores.values())

    if total_score > 0:
        weights = {name: score / total_score for name, score in scores.items()}
    else:
        # Equal weights if scores are not positive
        weights = {'xgb': 0.33, 'lgb': 0.33, 'cat': 0.34}

    # Calculate ensemble performance
    ensemble_pred = (
        xgb_pred * weights['xgb'] +
        lgb_pred * weights['lgb'] +
        cat_pred * weights['cat']
    )
    ensemble_score = np.corrcoef(ensemble_pred, y_val)[0, 1]

    print(f"\n{'='*80}")
    print(f"ENSEMBLE RESULTS for {factor_name.upper()}")
    print(f"{'='*80}")
    print(f"XGBoost:  {xgb_score:.4f} (weight: {weights['xgb']:.3f})")
    print(f"LightGBM: {lgb_score:.4f} (weight: {weights['lgb']:.3f})")
    print(f"CatBoost: {cat_score:.4f} (weight: {weights['cat']:.3f})")
    print(f"Ensemble: {ensemble_score:.4f}")
    print(f"{'='*80}")

    return {
        'models': models,
        'weights': weights,
        'val_score': ensemble_score,
        'individual_scores': scores
    }


def main():
    print("\n" + "="*80)
    print("MULTI-FACTOR MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    data_dir = Path(__file__).parent / "data"
    artifacts_dir = Path(__file__).parent / "artifacts"

    # Load training data
    print("\n[Step 1/4] Loading training data...")
    print("-" * 80)

    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")
    print(f"Loaded data: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Get only rows with labels (for training)
    df_train = df[df['label'].notna()].copy()
    print(f"Training data (with labels): {df_train.shape}")

    # Verify feature coverage
    print("\n[Step 2/4] Verifying factor definitions...")
    print("-" * 80)

    available_features = df_train.columns.tolist()
    coverage_report = verify_feature_coverage(available_features)

    for factor_name, info in coverage_report.items():
        status = "[OK]" if info['missing'] == 0 else "[WARNING]"
        print(f"{status} {factor_name}: {info['available']}/{info['total']} features "
              f"({info['coverage_pct']:.1f}%)")

        if info['missing'] > 0:
            print(f"    Missing: {info['missing_features']}")

    # Prepare target
    y = df_train['future_return'].values

    # Train/validation split (time-based)
    print("\n[Step 3/4] Train/validation split...")
    print("-" * 80)

    split_date = df_train['date'].quantile(0.8)
    train_mask = df_train['date'] < split_date
    val_mask = df_train['date'] >= split_date

    print(f"Training set: {train_mask.sum()} samples (before {split_date.date()})")
    print(f"Validation set: {val_mask.sum()} samples (from {split_date.date()})")

    y_train = y[train_mask]
    y_val = y[val_mask]

    # Train factor models
    print("\n[Step 4/4] Training factor-specific ensembles...")
    print("-" * 80)

    factor_ensembles = {}

    for factor_name, factor_features in FACTOR_GROUPS.items():
        # Filter to available features
        available_factor_features = [f for f in factor_features if f in available_features]

        if len(available_factor_features) == 0:
            print(f"\n[SKIP] {factor_name}: No features available")
            continue

        # Prepare data for this factor
        X = df_train[available_factor_features].values
        X = pd.DataFrame(X, columns=available_factor_features).ffill().fillna(0).values

        X_train_factor = X[train_mask]
        X_val_factor = X[val_mask]

        # Train ensemble for this factor
        ensemble = train_ensemble_for_factor(
            X_train_factor, y_train,
            X_val_factor, y_val,
            factor_name
        )

        # Save factor ensemble
        factor_ensembles[factor_name] = {
            'models': ensemble['models'],
            'weights': ensemble['weights'],
            'feature_cols': available_factor_features,
            'val_score': ensemble['val_score'],
            'individual_scores': ensemble['individual_scores']
        }

    # Save all factor ensembles
    print("\n" + "="*80)
    print("SAVING FACTOR MODELS")
    print("="*80)

    for factor_name, ensemble_data in factor_ensembles.items():
        output_file = artifacts_dir / f"ensemble_{factor_name}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(ensemble_data, f)

        print(f"[OK] Saved {factor_name}: {output_file}")
        print(f"     Features: {len(ensemble_data['feature_cols'])}")
        print(f"     Val Score: {ensemble_data['val_score']:.4f}")

    # Save factor weights config (with W1.5 effective weights)
    config_file = artifacts_dir / "factor_weights.json"
    import json

    # Compute factor scores
    factor_scores = {
        name: float(data['val_score'])
        for name, data in factor_ensembles.items()
    }

    # W1.5 (2026-05): also persist the IC-proportional "effective" weights so
    # downstream consumers (get_daily_signals_multi_factor, run_backtest_...)
    # don't have to recompute them and don't drift from training.
    from factors.factor_weighting import compute_ic_proportional
    effective_weights = compute_ic_proportional(factor_scores, floor=0.0)
    if effective_weights is None:
        print("\n[WARN] All factor ICs <= 0 -- effective_weights set to static "
              "(fallback engaged at signal time).")
        effective_weights = dict(FACTOR_WEIGHTS)
        weighting_strategy_used = "static_fallback"
    else:
        weighting_strategy_used = "ic_proportional"

    config = {
        'factor_weights_static': dict(FACTOR_WEIGHTS),   # hand-tuned, for audit only
        'effective_weights': effective_weights,          # what signals will use by default
        'weighting_strategy': weighting_strategy_used,
        'training_date': datetime.now().isoformat(),
        'factor_scores': factor_scores,
        # legacy alias so older readers keep working
        'factor_weights': dict(FACTOR_WEIGHTS),
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] Saved factor weights config: {config_file}")
    print(f"     Strategy used: {weighting_strategy_used}")

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    print(f"\nFactor Performance (Validation Correlation) -- "
          f"effective weights shown:")
    print("-" * 80)

    for factor_name in sorted(factor_ensembles.keys(),
                              key=lambda x: factor_ensembles[x]['val_score'],
                              reverse=True):
        score = factor_ensembles[factor_name]['val_score']
        eff_w = effective_weights.get(factor_name, 0.0)
        static_w = FACTOR_WEIGHTS.get(factor_name, 0)
        n_features = len(factor_ensembles[factor_name]['feature_cols'])

        marker = "  " if eff_w > 0 else "x "
        print(f"{marker}{factor_name:12} IC={score:>+7.4f}  "
              f"eff_w={eff_w*100:>5.1f}%  (static={static_w*100:>5.1f}%, "
              f"features={n_features:>2})")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext step: Run get_daily_signals_multi_factor.py to generate signals")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
