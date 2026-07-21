"""GDELT news factor experiment (research item B, 2026-07).

Builds trailing-only per-symbol news features from gdelt_daily.parquet
(daily average tone + article count, 2017+), adds them as a separate
`news` factor to the h=5 walk-forward, and reports the incremental IC.

Timing discipline:
- GDELT dates are calendar days. Signals are generated after the close,
  so day-T news is available for day-T predictions. Weekend/holiday news
  is attributed to the NEXT trading day (known before its close).
- All rolling features are trailing; no negative shifts.
- Rows before GDELT coverage (pre-2017) carry NaN -> 0.0 at matrix build;
  the per-fold inner validation naturally zero-weights the factor in
  folds where it has no information.

Usage:
    python evaluation/experiments/news_factor.py --max-folds 1   # smoke
    python evaluation/experiments/news_factor.py                 # full run
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

GDELT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gdelt_daily.parquet"

NEWS_FEATURES = [
    'news_tone', 'news_tone_chg_5d', 'news_tone_z60',
    'news_article_spike', 'news_tone_stability',
]
XS_NEWS_FEATURES = [f + '_xs' for f in NEWS_FEATURES]


def build_news_features(stock_dates: pd.Series) -> pd.DataFrame:
    """Aggregate GDELT calendar-day series onto trading days, then compute
    trailing features per symbol. Returns [date, symbol, news_*]."""
    g = pd.read_parquet(GDELT_PATH)
    g = g.dropna(subset=['gdelt_tone'], how='all')
    # BigQuery exports nullable extension dtypes (Float64Dtype) which numpy
    # can't interpret downstream; force plain float64.
    g['gdelt_tone'] = pd.to_numeric(g['gdelt_tone'], errors='coerce').astype('float64')
    g['gdelt_articles'] = pd.to_numeric(g['gdelt_articles'], errors='coerce').astype('float64')

    # Map each calendar day to the next trading day (>= that day).
    tdays = np.sort(stock_dates.dt.tz_localize(None).unique())
    cal = g['date'].to_numpy(dtype='datetime64[ns]')
    idx = np.searchsorted(tdays, cal, side='left')
    valid = idx < len(tdays)
    g = g[valid].copy()
    g['tdate'] = tdays[idx[valid]]

    agg = (g.groupby(['symbol', 'tdate'])
             .agg(news_tone=('gdelt_tone', 'mean'),
                  news_articles=('gdelt_articles', 'sum'))
             .reset_index().rename(columns={'tdate': 'date'})
             .sort_values(['symbol', 'date']))

    gb = agg.groupby('symbol', group_keys=False)
    agg['news_tone_chg_5d'] = gb['news_tone'].transform(lambda s: s - s.shift(5))
    roll_mean = gb['news_tone'].transform(lambda s: s.rolling(60, min_periods=20).mean())
    roll_std = gb['news_tone'].transform(lambda s: s.rolling(60, min_periods=20).std())
    agg['news_tone_z60'] = ((agg['news_tone'] - roll_mean)
                            / roll_std.replace(0, np.nan))
    art_avg = gb['news_articles'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    agg['news_article_spike'] = (agg['news_articles']
                                 / art_avg.replace(0, np.nan))
    agg['news_tone_stability'] = -gb['news_tone'].transform(
        lambda s: s.rolling(20, min_periods=10).std())

    return agg[['date', 'symbol'] + NEWS_FEATURES]


def main(max_folds: int | None = None):
    print("loading stock dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("building news features from GDELT...")
    news = build_news_features(df['date'])
    print(f"  news rows: {len(news):,}, symbols: {news['symbol'].nunique()}, "
          f"{news['date'].min().date()} .. {news['date'].max().date()}")

    # Align tz before merging.
    tz = df['date'].dt.tz
    if tz is not None:
        news['date'] = news['date'].dt.tz_localize(tz)
    df = df.merge(news, on=['date', 'symbol'], how='left')
    cov = df['news_tone'].notna().mean()
    cov_2017 = df.loc[df['date'] >= pd.Timestamp('2017-06-01', tz=tz),
                      'news_tone'].notna().mean()
    print(f"  coverage: {cov:.1%} overall, {cov_2017:.1%} since 2017-06")

    df = add_cross_sectional_zscore(df, NEWS_FEATURES, suffix='_xs',
                                    winsorize_pct=0.01)

    groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
    groups['news'] = NEWS_FEATURES + XS_NEWS_FEATURES

    run_walk_forward(
        WalkForwardConfig(horizon=5),
        max_folds=max_folds,
        df=df,
        factor_groups=groups,
        tag="newsN_ext",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    main(max_folds=args.max_folds)
