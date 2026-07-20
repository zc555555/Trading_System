"""Purged walk-forward evaluation of the multi-factor system (P1, 2026-07).

For each fold: train the 6 factor ensembles on a rolling window (with an
inner purged validation that sets base-model and factor weights), purge
``horizon + embargo`` days before the test window, then score the untouched
test window. Concatenating the non-overlapping test windows yields a fully
out-of-sample prediction panel for every date since the first fold.

Outputs (research/evaluation/results/):
    oos_predictions.parquet   per-(date,symbol) out-of-sample predictions
    walk_forward_report.json  fold metadata + per-factor/blended IC summaries
    fold_weights.csv          factor-weight history across folds

Hold-out convention: folds whose test window starts on/after HOLDOUT_START
are reported separately and must NEVER be used for tuning decisions.

Usage:
    python evaluation/purged_walk_forward.py            # full run
    python evaluation/purged_walk_forward.py --max-folds 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.factor_training import train_factor_ensembles, predict_panel  # noqa: E402
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stocks_with_time_windows.parquet"

HOLDOUT_START = "2025-07-01"


@dataclass
class WalkForwardConfig:
    train_window: int = 756      # ~3y rolling train window (trading days)
    test_window: int = 63        # ~1 quarter per fold
    step: int = 63               # == test_window -> non-overlapping OOS panel
    horizon: int = 1             # label horizon (days)
    embargo_days: int = 5        # extra gap beyond horizon at every boundary
    seed: int = 42


def generate_folds(dates: np.ndarray, cfg: WalkForwardConfig):
    """Yield (train_dates, test_dates) with a purge gap before each test.

    Train window: dates[i-train_window : i] minus its last
    (horizon + embargo) days. Test window: dates[i : i+test_window].
    A sample on the last kept train date has a label reaching
    ``horizon`` days forward -- still strictly inside the purge gap, so no
    train label overlaps any test date.
    """
    dates = np.sort(np.asarray(dates))
    purge = cfg.horizon + cfg.embargo_days
    folds = []
    i = cfg.train_window
    while i + cfg.test_window <= len(dates):
        train = dates[i - cfg.train_window: i - purge]
        test = dates[i: i + cfg.test_window]
        folds.append((train, test))
        i += cfg.step
    return folds


def run_walk_forward(cfg: WalkForwardConfig, max_folds: int | None = None,
                     data_path: Path = DATA_PATH,
                     df: pd.DataFrame | None = None,
                     factor_groups: dict | None = None,
                     tag: str = "") -> dict:
    """Run the walk-forward. Experiments can pass a pre-augmented ``df``,
    custom ``factor_groups``, and a ``tag`` appended to result filenames;
    the default call evaluates the production factor set."""
    t0 = time.time()
    print("=" * 78)
    print(f"PURGED WALK-FORWARD EVALUATION  (label horizon = {cfg.horizon}d"
          + (f", variant '{tag}'" if tag else "") + ")")
    print("=" * 78)
    print(f"config: {asdict(cfg)}")

    if df is None:
        df = pd.read_parquet(data_path)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Compute the horizon label on the fly from close prices; the stored
    # `future_return` column is 1-day only. Same formula the labels have
    # always used: log(close[t+h] / close[t]) per symbol.
    target_col = f'future_return_{cfg.horizon}d'
    df[target_col] = df.groupby('symbol')['close'].transform(
        lambda x: np.log(x.shift(-cfg.horizon) / x))
    if cfg.horizon == 1 and 'future_return' in df.columns:
        both = df[['future_return', target_col]].dropna()
        max_diff = float((both['future_return'] - both[target_col]).abs().max())
        print(f"sanity: on-the-fly 1d label vs stored future_return, "
              f"max diff = {max_diff:.2e}")

    labeled = df[df[target_col].notna()]
    print(f"data: {df.shape[0]:,} rows, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}, "
          f"{df['symbol'].nunique()} symbols")

    folds = generate_folds(labeled['date'].unique(), cfg)
    if max_folds:
        folds = folds[:max_folds]
    print(f"folds: {len(folds)} "
          f"(purge {cfg.horizon + cfg.embargo_days}d before every test window)")

    panels, fold_meta, weight_rows = [], [], []
    for k, (train_dates, test_dates) in enumerate(folds, 1):
        f0 = time.time()
        print(f"\n--- fold {k}/{len(folds)}  "
              f"train {pd.Timestamp(train_dates[0]).date()}"
              f"..{pd.Timestamp(train_dates[-1]).date()}  "
              f"test {pd.Timestamp(test_dates[0]).date()}"
              f"..{pd.Timestamp(test_dates[-1]).date()}")

        train_df = labeled[labeled['date'].isin(train_dates)]
        test_df = df[df['date'].isin(test_dates)]

        fold = train_factor_ensembles(
            train_df, horizon=cfg.horizon, embargo_days=cfg.embargo_days,
            target_col=target_col, factor_groups=factor_groups, seed=cfg.seed)

        panel = predict_panel(fold, test_df, target_col=target_col)
        panel['fold'] = k
        panels.append(panel)

        weight_rows.append({'fold': k,
                            'test_start': str(pd.Timestamp(test_dates[0]).date()),
                            **fold['factor_weights']})
        fold_meta.append({
            'fold': k,
            'train_start': str(pd.Timestamp(train_dates[0]).date()),
            'train_end': str(pd.Timestamp(train_dates[-1]).date()),
            'test_start': str(pd.Timestamp(test_dates[0]).date()),
            'test_end': str(pd.Timestamp(test_dates[-1]).date()),
            'factor_scores': fold['factor_scores'],
            'factor_weights': fold['factor_weights'],
        })
        print(f"    weights: " + "  ".join(
            f"{f}={w:.2f}" for f, w in sorted(fold['factor_weights'].items(),
                                              key=lambda kv: -kv[1])))
        print(f"    fold time: {time.time() - f0:.0f}s")

    panel = pd.concat(panels, ignore_index=True)

    # ---- IC report (out-of-sample, per-date rank IC) --------------------
    report = {'config': asdict(cfg), 'generated_at': datetime.now().isoformat(),
              'holdout_start': HOLDOUT_START, 'n_folds': len(folds),
              'folds': fold_meta, 'ic': {}}

    holdout_mask = panel['date'] >= pd.Timestamp(HOLDOUT_START).tz_localize(
        panel['date'].dt.tz) if panel['date'].dt.tz else panel['date'] >= HOLDOUT_START

    # Overlapping h-day labels autocorrelate consecutive daily ICs;
    # Newey-West with h-1 lags keeps the t-stat honest.
    nw_lags = cfg.horizon - 1
    pred_cols = ['pred'] + [c for c in panel.columns if c.startswith('factor_')]
    print("\n" + "=" * 78)
    print(f"OUT-OF-SAMPLE DAILY RANK IC  (horizon {cfg.horizon}d, "
          f"NW lags {nw_lags}, dev < {HOLDOUT_START} <= holdout)")
    print("=" * 78)
    print(f"{'signal':<20}{'segment':<10}{'mean IC':>9}{'t-stat':>8}"
          f"{'IC IR':>7}{'%>0':>7}{'days':>6}")
    for col in pred_cols:
        report['ic'][col] = {}
        for seg_name, seg in (('dev', panel[~holdout_mask]),
                              ('holdout', panel[holdout_mask]),
                              ('all', panel)):
            s = summarize_ic(daily_rank_ic(seg, col, target_col=target_col),
                             nw_lags=nw_lags)
            report['ic'][col][seg_name] = s
            if s['n_days']:
                print(f"{col:<20}{seg_name:<10}{s['ic_mean']:>+9.4f}"
                      f"{s['t_stat']:>8.2f}{s['ic_ir']:>7.3f}"
                      f"{s['pct_positive']:>7.1%}{s['n_days']:>6}")

    # ---- persist (files suffixed by horizon + variant tag) --------------
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_h{cfg.horizon}" + (f"_{tag}" if tag else "")
    report['target_col'] = target_col
    panel.to_parquet(RESULTS_DIR / f"oos_predictions{suffix}.parquet", index=False)
    with open(RESULTS_DIR / f"walk_forward_report{suffix}.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    pd.DataFrame(weight_rows).to_csv(RESULTS_DIR / f"fold_weights{suffix}.csv",
                                     index=False)

    print(f"\nsaved: {RESULTS_DIR / f'oos_predictions{suffix}.parquet'}")
    print(f"saved: {RESULTS_DIR / f'walk_forward_report{suffix}.json'}")
    print(f"total time: {(time.time() - t0) / 60:.1f} min")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None,
                    help="Limit folds (smoke test)")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20],
                    help="Label horizon in trading days")
    ap.add_argument("--tag", type=str, default="",
                    help="Variant tag appended to result filenames")
    args = ap.parse_args()
    run_walk_forward(WalkForwardConfig(horizon=args.horizon),
                     max_folds=args.max_folds, tag=args.tag)
