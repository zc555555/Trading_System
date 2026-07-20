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

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from factors.factor_definitions import FACTOR_GROUPS, FACTOR_WEIGHTS, verify_feature_coverage
from factors.factor_weighting import compute_ic_proportional
from evaluation.factor_training import train_factor_ensembles, refit_on_full_data

HORIZON = 1
EMBARGO_DAYS = 5


def main():
    print("\n" + "=" * 80)
    print("MULTI-FACTOR MODEL TRAINING  (rank-IC weighted, purged inner val)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path(__file__).parent / "data"
    artifacts_dir = Path(__file__).parent / "artifacts"

    print("\n[Step 1/4] Loading training data...")
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")
    df_train = df[df['label'].notna()].copy()
    print(f"  {df_train.shape[0]:,} labeled rows, "
          f"{df_train['date'].min().date()} .. {df_train['date'].max().date()}, "
          f"{df_train['symbol'].nunique()} symbols")

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
        df_train, horizon=HORIZON, embargo_days=EMBARGO_DAYS)

    print("\n[Step 4/4] Refitting base models on all labeled data...")
    fold = refit_on_full_data(fold, df_train)

    # ---- persist ensembles (schema-compatible with all consumers) -------
    print("\n" + "=" * 80)
    print("SAVING FACTOR MODELS")
    print("=" * 80)
    for factor_name, ens in fold['ensembles'].items():
        payload = {
            'models': ens['models'],
            'weights': ens['weights'],
            'feature_cols': ens['feature_cols'],
            'val_score': ens['factor_ic'],          # now: mean daily rank IC
            'individual_scores': ens['model_ics'],  # now: per-model rank ICs
        }
        out = artifacts_dir / f"ensemble_{factor_name}.pkl"
        with open(out, 'wb') as f:
            pickle.dump(payload, f)
        print(f"  [OK] {factor_name}: rankIC={ens['factor_ic']:+.4f} -> {out.name}")

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
        # legacy alias so older readers keep working
        'factor_weights': dict(FACTOR_WEIGHTS),
    }
    with open(artifacts_dir / "factor_weights.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n[OK] Saved factor_weights.json (strategy={strategy_used})")

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
