"""Residualized-label experiment (Layer 3, item 1).

Hypothesis: predicting RAW h-day returns wastes model capacity on the
market component (which no cross-sectional book can monetize and which
the construction grid showed leaking in as beta). Training on
market-residualized returns should sharpen the cross-sectional ranking.

Variants at the production horizon (h=20):
  A  demarket   y_i = fret_i - fret_eqw          (beta = 1 for everyone)
  B  debeta     y_i = fret_i - beta_i * fret_eqw (beta_i = trailing 120d
                 OLS beta of daily returns vs the equal-weight universe
                 return; PAST data only)

fret_eqw is the per-date cross-sectional mean of the same forward
window -- inside the label horizon, so the existing purge/embargo
geometry still holds. Betas are strictly trailing.

Verdict metric: daily rank IC of the blended prediction against the RAW
forward return (what a book actually monetizes), compared with the raw-
label baseline (walk_forward_report_h20_ext) on the standard four
evidence tiers.

Usage:
    python evaluation/experiments/residual_labels.py --variant A --max-folds 1
    python evaluation/experiments/residual_labels.py            # A then B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.purged_walk_forward import (  # noqa: E402
    WalkForwardConfig, run_walk_forward, DATA_PATH, RESULTS_DIR)
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402

HORIZON = 20
SEGMENTS = [
    ("virgin_early", None, "2021-12-30"),
    ("seen_dev", "2021-12-30", "2025-07-01"),
    ("holdout", "2025-07-01", "2026-05-16"),
    ("fresh", "2026-05-16", None),
]


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol', group_keys=False)

    raw = f'future_return_{HORIZON}d'
    df[raw] = g['close'].transform(
        lambda x: np.log(x.shift(-HORIZON) / x))

    # Equal-weight market forward return over the SAME window, per date.
    df['_fret_mkt'] = df.groupby('date')[raw].transform('mean')

    # Variant A: unit-beta demarket.
    df['resid_A'] = df[raw] - df['_fret_mkt']

    # Variant B: trailing 120d beta vs the equal-weight universe daily return.
    df['_ret'] = g['close'].transform(lambda s: s.pct_change())
    df['_mkt'] = df.groupby('date')['_ret'].transform('mean')
    g = df.groupby('symbol', group_keys=False)
    cov = g.apply(lambda d: d['_ret'].rolling(120, min_periods=60)
                  .cov(d['_mkt'])).reset_index(drop=True)
    var = g.apply(lambda d: d['_mkt'].rolling(120, min_periods=60)
                  .var()).reset_index(drop=True)
    df['_beta'] = (cov / var.replace(0, np.nan)).clip(-1.0, 3.0)
    df['resid_B'] = df[raw] - df['_beta'] * df['_fret_mkt']
    # No trailing beta yet (warm-up) -> fall back to unit beta.
    df['resid_B'] = df['resid_B'].fillna(df['resid_A'])

    return df.drop(columns=['_ret', '_mkt'])


def verdict(panel_tag: str, base_report_panel: str = "oos_predictions_h20_ext.parquet"):
    raw_col = f'future_return_{HORIZON}d'
    base = pd.read_parquet(RESULTS_DIR / base_report_panel)
    var = pd.read_parquet(RESULTS_DIR / f"oos_predictions_h20_{panel_tag}.parquet")
    # variant panel carries the residual target; merge the raw target in
    if raw_col not in var.columns:
        var = var.merge(base[['date', 'symbol', raw_col]],
                        on=['date', 'symbol'], how='left')

    print(f"\n{'segment':<14}{'baseline pred~raw':>22}{'residual pred~raw':>22}")
    print("-" * 60)
    rows = []
    for name, start, end in SEGMENTS:
        def seg(df):
            tz = df['date'].dt.tz
            m = pd.Series(True, index=df.index)
            if start:
                m &= df['date'] >= pd.Timestamp(start).tz_localize(tz)
            if end:
                m &= df['date'] < pd.Timestamp(end).tz_localize(tz)
            return df[m]
        b = summarize_ic(daily_rank_ic(seg(base), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        v = summarize_ic(daily_rank_ic(seg(var), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)

        def fmt(s):
            return (f"{s['ic_mean']:+.4f} (t={s['t_stat']:.2f})"
                    if s['n_days'] else "n/a")
        print(f"{name:<14}{fmt(b):>22}{fmt(v):>22}")
        rows.append({"segment": name, "base_ic": b['ic_mean'],
                     "base_t": b['t_stat'], "var_ic": v['ic_mean'],
                     "var_t": v['t_stat']})
    out = RESULTS_DIR / f"residual_labels_verdict_{panel_tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


def main(variants: list[str], max_folds: int | None):
    print("building residual labels...")
    df = pd.read_parquet(DATA_PATH)
    df = build_labels(df)
    for v in variants:
        print(f"\n[variant {v}] beta coverage: "
              f"{df['_beta'].notna().mean():.1%}" if v == 'B' else f"\n[variant {v}]")
        run_walk_forward(
            WalkForwardConfig(horizon=HORIZON),
            max_folds=max_folds,
            df=df,
            target_override=f'resid_{v}',
            tag=f"res{v}",
        )
        if not max_folds:
            verdict(f"res{v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B", "both"], default="both")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    main(["A", "B"] if args.variant == "both" else [args.variant],
         max_folds=args.max_folds)
