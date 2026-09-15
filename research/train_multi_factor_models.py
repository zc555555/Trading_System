"""
MULTI-FACTOR MODEL TRAINING (P1 rebuild, 2026-07)

Trains one XGB + LGBM + CatBoost ensemble per factor via the shared core in
evaluation/factor_training.py -- the same code the purged walk-forward uses,
so production training and evaluation can never drift.

What changed vs the pre-P1 version:
- Factor/model scores are per-date cross-sectional rank ICs on a PURGED
  inner validation (horizon + 5-day embargo), not pooled Pearson on an
  unpurged 80/20 split.
- Base-model blend weights clip negative ICs to zero (previously a model
  with negative validation correlation received a NEGATIVE weight).
- After weights are measured, base models are REFIT on all labeled data so
  production predicts with the freshest possible fit.

Output format of artifacts/ensemble_{factor}.pkl is unchanged
({models, weights, feature_cols, val_score, individual_scores}), so
get_daily_signals_multi_factor.py and the backtest load them as before.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from factors.factor_definitions import FACTOR_GROUPS, FACTOR_WEIGHTS, verify_feature_coverage
from factors.factor_weighting import compute_ic_proportional
from evaluation.factor_training import train_factor_ensembles, refit_on_full_data

# 2026-07 (D): production trains at the 20-day horizon to match the live
# 20-day staggered hold -- the only configuration the honest walk-forward
# found near breakeven after costs. The stored parquet label is 1-day, so
# the 20d target is computed on the fly below (same formula as the
# walk-forward evaluation).
HORIZON = 20
EMBARGO_DAYS = 5

# 2026-09-14 (review item 3 / decision 1): production trains exactly the way
# purged_walk_forward evaluates a fold -- the LAST TRAIN_WINDOW labeled
# sessions, inner purged validation for base-model and factor weights, and
# no refit on the full window. Every reported IC / Sharpe describes this
# policy; the old full-history + refit models were never evaluated.
TRAIN_WINDOW = 756            # == evaluation.purged_walk_forward FoldConfig.train_window
REFIT_ON_FULL = False


def restrict_to_window(df_train, n_sessions: int):
    """Keep the last `n_sessions` distinct dates of a labeled frame."""
    if not n_sessions:
        return df_train
    dates = np.sort(df_train["date"].unique())
    keep = dates[-int(n_sessions):]
    return df_train[df_train["date"].isin(keep)].copy()



def main():
    print("\n" + "=" * 80)
    print("MULTI-FACTOR MODEL TRAINING  (rank-IC weighted, purged inner val)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path(__file__).parent / "data"
    artifacts_dir = Path(__file__).parent / "artifacts"

    print("\n[Step 1/4] Loading training data...")
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    target_col = f'future_return_{HORIZON}d'
    df = df.sort_values(['symbol', 'date'])
    df[target_col] = df.groupby('symbol')['close'].transform(
        lambda x: np.log(x.shift(-HORIZON) / x))
    df_train = df[df[target_col].notna()].copy()
    print(f"  {df_train.shape[0]:,} rows with {HORIZON}d labels, "
          f"{df_train['date'].min().date()} .. {df_train['date'].max().date()}, "
          f"{df_train['symbol'].nunique()} symbols")
    df_train = restrict_to_window(df_train, TRAIN_WINDOW)
    print(f"  rolling window: last {TRAIN_WINDOW} sessions -> {df_train.shape[0]:,} rows, "
          f"{df_train['date'].min().date()} .. {df_train['date'].max().date()} (the evaluated policy)")

    print("\n[Step 2/4] Verifying factor definitions...")
    coverage = verify_feature_coverage(df_train.columns.tolist())
    for factor_name, info in coverage.items():
        status = "[OK]" if info['missing'] == 0 else "[WARNING]"
        print(f"  {status} {factor_name}: {info['available']}/{info['total']} features")
        if info['missing'] > 0:
            print(f"      Missing: {info['missing_features']}")

    print(f"\n[Step 3/4] Training with purged inner validation "
          f"(horizon={HORIZON}, embargo={EMBARGO_DAYS})...")
    fold = train_factor_ensembles(
        df_train, horizon=HORIZON, embargo_days=EMBARGO_DAYS,
        target_col=target_col)

    if REFIT_ON_FULL:
        print("\n[Step 4/4] Refitting base models on the whole window...")
        fold = refit_on_full_data(fold, df_train, target_col=target_col)
    else:
        print("\n[Step 4/4] No refit: the inner-validation fold models go to production, "
              "exactly as the walk-forward evaluates them")

    # ---- persist ensembles (schema-compatible with all consumers) -------
    print("\n" + "=" * 80)
    print("SAVING FACTOR MODELS")
    print("=" * 80)
    # every ensemble goes to staging first; the set is promoted as a whole
    # (factors/model_release.py) so a crash mid-way never leaves a mixed set
    from factors import model_release
    staging = model_release.stage_dir(artifacts_dir)
    staging.mkdir(parents=True, exist_ok=True)
    for factor_name, ens in fold['ensembles'].items():
        payload = {
            'models': ens['models'],
            'weights': ens['weights'],
            'feature_cols': ens['feature_cols'],
            'val_score': ens['factor_ic'],          # now: mean daily rank IC
            'individual_scores': ens['model_ics'],  # now: per-model rank ICs
        }
        out = staging / f"ensemble_{factor_name}.pkl"
        with open(out, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  [OK] {factor_name}: rankIC={ens['factor_ic']:+.4f} -> staging/{out.name}")
    model_release.promote(artifacts_dir, list(fold['ensembles'].keys()))
    print(f"  [OK] promoted {len(fold['ensembles'])} ensembles from staging")

    # ---- factor weights json --------------------------------------------
    factor_scores = fold['factor_scores']
    effective = compute_ic_proportional(factor_scores, floor=0.0, power=0.5)
    strategy_used = "ic_sqrt"
    if effective is None:
        print("\n[WARN] All factor rank ICs <= 0 -- falling back to static weights.")
        effective = dict(FACTOR_WEIGHTS)
        strategy_used = "static_fallback"

    config = {
        'factor_weights_static': dict(FACTOR_WEIGHTS),
        'effective_weights': effective,
        'weighting_strategy': strategy_used,
        'training_date': datetime.now().isoformat(),
        'factor_scores': factor_scores,
        # P1 provenance: how these scores were measured
        'score_metric': 'daily_cross_sectional_rank_ic',
        'inner_split': fold['inner_split'],
        'horizon': HORIZON,
        'train_window_sessions': TRAIN_WINDOW,
        'train_start': str(pd.to_datetime(df_train['date']).min().date()),
        'train_end': str(pd.to_datetime(df_train['date']).max().date()),
        'refit_on_full': REFIT_ON_FULL,
        # legacy alias so older readers keep working
        'factor_weights': dict(FACTOR_WEIGHTS),
    }
    with open(artifacts_dir / "factor_weights.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[OK] Saved factor_weights.json (strategy={strategy_used})")
    manifest = model_release.write_manifest(artifacts_dir, list(fold['ensembles'].keys()), horizon=HORIZON,
                                            data_max_date=str(pd.to_datetime(df_train['date']).max().date()),
                                            extra={"train_window_sessions": TRAIN_WINDOW,
                                                   "train_start": str(pd.to_datetime(df_train['date']).min().date()),
                                                   "refit_on_full": REFIT_ON_FULL})
    print(f"[OK] model_release.json written: release {manifest['release_id']}")

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY  (per-date rank IC on purged inner validation)")
    print("=" * 80)
    for name in sorted(factor_scores, key=factor_scores.get, reverse=True):
        w = effective.get(name, 0.0)
        marker = "  " if w > 0 else "x "
        print(f"{marker}{name:12} rankIC={factor_scores[name]:>+8.4f}  "
              f"weight={w * 100:>5.1f}%")

    print("\n" + "=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Next step: get_daily_signals_multi_factor.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
