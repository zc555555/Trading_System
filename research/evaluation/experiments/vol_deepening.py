"""Volatility-factor deepening experiment (research item #3, 2026-07).

Hypothesis: the volatility factor (the only significant honest signal,
rankIC@5d +0.0177 t=2.06) is measured with close-to-close vol only. OHLC
range estimators (Garman-Klass, Parkinson), idiosyncratic vol (the classic
low-vol-anomaly variable), term structure, vol-of-vol, downside semivol and
an EWMA forecast add genuinely new measurement -- do they raise the factor's
IC at the h=5 horizon?

Variants (17-fold purged walk-forward each, h=5):
    A  baseline            -- saved report walk_forward_report_h5.json
    B  vol_aug             -- volatility factor list AUGMENTED with new features
    C  vol_ext             -- new features as a SEPARATE 7th factor,
                              original volatility untouched (clean attribution)

All new features are strictly trailing (rolling windows on past data only);
the leakage-guard test scans this file too via its use of standard ops.

Usage:
    python evaluation/experiments/vol_deepening.py            # B then C
    python evaluation/experiments/vol_deepening.py --variant B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from factors.factor_definitions import FACTOR_GROUPS  # noqa: E402
from features.cross_sectional import add_cross_sectional_zscore  # noqa: E402
from evaluation.purged_walk_forward import (  # noqa: E402
    WalkForwardConfig, run_walk_forward, DATA_PATH)

NEW_VOL_FEATURES = [
    'gk_vol_20', 'parkinson_vol_20', 'idio_vol_20', 'vol_term_structure',
    'vol_of_vol_20', 'downside_semivol_20', 'ewma_vol', 'vol_premium_20',
]
XS_VOL_FEATURES = [f + '_xs' for f in NEW_VOL_FEATURES]


def compute_new_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the new volatility features (all trailing-only)."""
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    g = df.groupby('symbol', group_keys=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_hl = np.log(df['high'] / df['low']).replace([np.inf, -np.inf], np.nan)
        log_co = np.log(df['close'] / df['open']).replace([np.inf, -np.inf], np.nan)

    # Garman-Klass and Parkinson range estimators (per-day variance terms,
    # rolling 20d mean, annualized vol). Range estimators use intraday
    # high/low information that close-to-close vol cannot see.
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    pk_daily = log_hl ** 2 / (4 * np.log(2))
    df['_gk'] = gk_daily
    df['_pk'] = pk_daily
    g = df.groupby('symbol', group_keys=False)
    df['gk_vol_20'] = np.sqrt(
        g['_gk'].transform(lambda s: s.rolling(20, min_periods=10).mean().clip(lower=0)) * 252)
    df['parkinson_vol_20'] = np.sqrt(
        g['_pk'].transform(lambda s: s.rolling(20, min_periods=10).mean().clip(lower=0)) * 252)

    # Daily returns and an equal-weight market proxy (no dependency on the
    # _y market-column naming).
    df['_ret'] = g['close'].transform(lambda s: s.pct_change())
    mkt = df.groupby('date')['_ret'].transform('mean')
    df['_resid'] = df['_ret'] - mkt

    g = df.groupby('symbol', group_keys=False)
    # Idiosyncratic vol: 20d std of market-residual returns (low-vol anomaly).
    df['idio_vol_20'] = g['_resid'].transform(
        lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(252)

    # Term structure: short vol relative to long vol.
    vol5 = g['_ret'].transform(lambda s: s.rolling(5, min_periods=3).std())
    vol60 = g['_ret'].transform(lambda s: s.rolling(60, min_periods=30).std())
    df['vol_term_structure'] = (vol5 / vol60).replace([np.inf, -np.inf], np.nan)

    # Vol-of-vol: 20d std of the rolling 5d vol series.
    df['_vol5'] = vol5
    g = df.groupby('symbol', group_keys=False)
    df['vol_of_vol_20'] = g['_vol5'].transform(
        lambda s: s.rolling(20, min_periods=10).std())

    # Downside semivol: sqrt(mean(min(r,0)^2)) over 20d.
    df['_neg2'] = np.minimum(df['_ret'], 0.0) ** 2
    g = df.groupby('symbol', group_keys=False)
    df['downside_semivol_20'] = np.sqrt(
        g['_neg2'].transform(lambda s: s.rolling(20, min_periods=10).mean()) * 252)

    # RiskMetrics EWMA vol (lambda=0.94) -- a cheap GARCH-lite forecast.
    df['_ret2'] = df['_ret'] ** 2
    g = df.groupby('symbol', group_keys=False)
    df['ewma_vol'] = np.sqrt(
        g['_ret2'].transform(lambda s: s.ewm(alpha=0.06, min_periods=10).mean()) * 252)

    # Vol premium: range-based vol minus close-based vol -- gap between the
    # two estimators carries information about intraday vs overnight risk.
    close_vol20 = g['_ret'].transform(
        lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(252)
    df['vol_premium_20'] = df['gk_vol_20'] - close_vol20

    df = df.drop(columns=['_gk', '_pk', '_ret', '_resid', '_vol5', '_neg2', '_ret2'])

    # Cross-sectional variants (per-date z-scores; same-day only, no leak).
    df = add_cross_sectional_zscore(df, NEW_VOL_FEATURES, suffix='_xs',
                                    winsorize_pct=0.01)
    # add_cross_sectional_zscore may leave NaN on thin dates; keep NaN -- the
    # training core converts to 0.0 at matrix build, matching the pipeline.
    return df


def variant_factor_groups(variant: str) -> dict:
    groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
    if variant == 'B':
        groups['volatility'] = groups['volatility'] + NEW_VOL_FEATURES + XS_VOL_FEATURES
    elif variant == 'C':
        groups['vol_ext'] = NEW_VOL_FEATURES + XS_VOL_FEATURES
    else:
        raise ValueError(variant)
    return groups


def main(variants: list[str], max_folds: int | None = None, tag_suffix: str = ""):
    print("loading + augmenting dataset...")
    df = pd.read_parquet(DATA_PATH)
    df = compute_new_vol_features(df)
    n_valid = df[NEW_VOL_FEATURES].notna().mean()
    print("new feature coverage:")
    for f in NEW_VOL_FEATURES:
        print(f"  {f:<22} {n_valid[f]:.1%}")

    for variant in variants:
        run_walk_forward(
            WalkForwardConfig(horizon=5),
            max_folds=max_folds,
            df=df,
            factor_groups=variant_factor_groups(variant),
            tag=f"vol{variant}{tag_suffix}",
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["B", "C", "both"], default="both")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--tag-suffix", type=str, default="",
                    help="Appended to the variant tag (e.g. '_ext')")
    args = ap.parse_args()
    main(["B", "C"] if args.variant == "both" else [args.variant],
         max_folds=args.max_folds, tag_suffix=args.tag_suffix)
