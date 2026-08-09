"""PEAD (post-earnings announcement drift) factor experiment (Layer 4).

The most robust documented anomaly reachable with free data: stocks
drift in the direction of their earnings surprise for weeks after the
announcement -- a natural fit for the 20-day book.

Features (all trailing-safe; announcement info becomes usable at the
SECOND session at-or-after the announcement date, which covers the
before-open/after-close timing ambiguity):

    days_since_earn   sessions since last announcement (capped 90)
    last_surprise     EPS surprise % of last announcement (clipped +-100)
    surprise_decay    last_surprise * exp(-days_since/20)  (the drift core)
    ear_react         close[a+1]/close[a-1]-1 announcement reaction
    react_decay       ear_react * exp(-days_since/20)

plus cross-sectional z-scores of the two decay features. Added as a
7th `pead` factor at h=20 on the extended data; verdict = four-tier
rank IC of the blend vs the raw forward return against the h20_ext
baseline.

Usage:
    python evaluation/experiments/pead_factor.py --max-folds 1   # smoke
    python evaluation/experiments/pead_factor.py
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
    WalkForwardConfig, run_walk_forward, DATA_PATH, RESULTS_DIR)
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402

EARNINGS = Path(__file__).resolve().parent.parent.parent / "data" / "earnings_dates.parquet"
HORIZON = 20
PEAD_FEATURES = ['days_since_earn', 'last_surprise', 'surprise_decay',
                 'ear_react', 'react_decay']
XS_FEATURES = ['surprise_decay_xs', 'react_decay_xs']
SEGMENTS = [
    ("virgin_early", None, "2021-12-30"),
    ("seen_dev", "2021-12-30", "2025-07-01"),
    ("holdout", "2025-07-01", "2026-05-16"),
    ("fresh", "2026-05-16", None),
]


def build_pead_features(df: pd.DataFrame) -> pd.DataFrame:
    ann = pd.read_parquet(EARNINGS)
    ann['ann_date'] = pd.to_datetime(ann['earnings_ts']).dt.tz_localize(None).dt.normalize()
    ann = (ann.dropna(subset=['ann_date'])
              .sort_values(['symbol', 'ann_date'])
              .drop_duplicates(['symbol', 'ann_date']))

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    sessions = np.sort(df['date'].dt.tz_localize(None).unique())
    sess_ord = {d: i for i, d in enumerate(sessions)}

    # Announcement reaction: close[a+1] / close[a-1] - 1 per symbol.
    px = df[['symbol', 'date', 'close']].copy()
    px['snorm'] = px['date'].dt.tz_localize(None)
    px['ord'] = px['snorm'].map(sess_ord)
    close_by_ord = px.set_index(['symbol', 'ord'])['close']

    # Map each announcement to session ordinals.
    ann_naive = ann['ann_date'].to_numpy(dtype='datetime64[ns]')
    idx = np.searchsorted(sessions, ann_naive, side='left')
    ann['a_ord'] = np.clip(idx, 0, len(sessions) - 1)
    # usable from the SECOND session at-or-after the announcement
    ann['eff_ord'] = ann['a_ord'] + 1

    def _react(row):
        try:
            c_prev = close_by_ord.get((row['symbol'], row['a_ord'] - 1))
            c_next = close_by_ord.get((row['symbol'], row['a_ord'] + 1))
            if c_prev and c_next and c_prev > 0:
                return c_next / c_prev - 1
        except Exception:
            pass
        return np.nan
    ann['ear_react'] = ann.apply(_react, axis=1)
    ann['surprise_pct'] = pd.to_numeric(ann['surprise_pct'],
                                        errors='coerce').clip(-100, 100)

    # As-of merge on session ordinal: last announcement whose eff_ord <= ord.
    df['snorm'] = df['date'].dt.tz_localize(None)
    df['ord'] = df['snorm'].map(sess_ord)
    merged = pd.merge_asof(
        df[['symbol', 'ord']].reset_index().sort_values('ord'),
        ann[['symbol', 'eff_ord', 'a_ord', 'surprise_pct', 'ear_react']]
            .sort_values('eff_ord'),
        left_on='ord', right_on='eff_ord', by='symbol',
        direction='backward').set_index('index').sort_index()

    days_since = (merged['ord'] - merged['a_ord']).clip(upper=90)
    decay = np.exp(-days_since / 20.0)
    df['days_since_earn'] = days_since.fillna(90.0)
    df['last_surprise'] = merged['surprise_pct'].fillna(0.0)
    df['surprise_decay'] = (merged['surprise_pct'] * decay).fillna(0.0)
    df['ear_react'] = merged['ear_react'].fillna(0.0)
    df['react_decay'] = (merged['ear_react'] * decay).fillna(0.0)
    df = df.drop(columns=['snorm', 'ord'])

    df = add_cross_sectional_zscore(df, ['surprise_decay', 'react_decay'],
                                    suffix='_xs', winsorize_pct=0.01)
    return df


def verdict(tag: str):
    raw_col = f'future_return_{HORIZON}d'
    base = pd.read_parquet(RESULTS_DIR / "oos_predictions_h20_ext.parquet")
    var = pd.read_parquet(RESULTS_DIR / f"oos_predictions_h20_{tag}.parquet")

    print(f"\n{'segment':<14}{'baseline':>20}{'with pead':>20}{'pead alone':>20}")
    print("-" * 76)
    rows = []
    for name, start, end in SEGMENTS:
        def seg(d):
            tz = d['date'].dt.tz
            m = pd.Series(True, index=d.index)
            if start:
                m &= d['date'] >= pd.Timestamp(start).tz_localize(tz)
            if end:
                m &= d['date'] < pd.Timestamp(end).tz_localize(tz)
            return d[m]
        b = summarize_ic(daily_rank_ic(seg(base), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        v = summarize_ic(daily_rank_ic(seg(var), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        f = summarize_ic(daily_rank_ic(seg(var), 'factor_pead',
                                       target_col=raw_col),
                         nw_lags=HORIZON - 1)

        def fmt(s):
            return (f"{s['ic_mean']:+.4f} (t={s['t_stat']:.2f})"
                    if s['n_days'] else "n/a")
        print(f"{name:<14}{fmt(b):>20}{fmt(v):>20}{fmt(f):>20}")
        rows.append({"segment": name, "base_ic": b['ic_mean'],
                     "with_ic": v['ic_mean'], "pead_ic": f['ic_mean'],
                     "base_t": b['t_stat'], "with_t": v['t_stat'],
                     "pead_t": f['t_stat']})
    out = RESULTS_DIR / f"pead_verdict_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


def main(max_folds: int | None, tail: int = 0):
    print("loading dataset + building PEAD features...")
    df = pd.read_parquet(DATA_PATH)
    df = build_pead_features(df)
    has_ann = (df['days_since_earn'] < 90).mean()
    print(f"coverage: {has_ann:.1%} of rows within 90 sessions of an announcement")

    groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
    groups['pead'] = PEAD_FEATURES + XS_FEATURES

    run_walk_forward(
        WalkForwardConfig(horizon=HORIZON, min_tail_test=tail),
        max_folds=max_folds,
        df=df,
        factor_groups=groups,
        tag="peadP",
    )
    if not max_folds:
        verdict("peadP")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--tail", type=int, default=0)
    args = ap.parse_args()
    main(max_folds=args.max_folds, tail=args.tail)
