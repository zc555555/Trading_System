"""
Cross-sectional feature engineering.

Converts per-stock absolute features into per-date relative rankings.
This is the standard approach used by every cross-sectional alpha shop
(AQR, Two Sigma, Worldquant) for blending across a universe of stocks.

Why
---
A factor model that tries to predict cross-sectional returns from
absolute features like ``price_vs_ma_30d = +5%`` cannot tell whether
+5% above MA is "good for this stock" or "bad relative to peers today
where the median stock is +8% above MA". Z-scoring within each date
solves this: positive z-score means the stock is above its peers
on that feature today, negative means it is below them.

Functions
---------
add_cross_sectional_zscore(df, features, suffix='_xs', winsorize_pct=0.01)
    For each (date, feature), compute (x - mean) / std across the cross
    section. Winsorizes the tails first so a single bad print does not
    dominate the standardization.

add_cross_sectional_rank(df, features, suffix='_xrank')
    For each (date, feature), compute the percentile rank in [0, 1].
    Robust to outliers, throws away magnitude.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    """Clip series to [pct, 1-pct] quantiles. Returns a new series."""
    if pct <= 0:
        return s
    lo = s.quantile(pct)
    hi = s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def add_cross_sectional_zscore(
    df: pd.DataFrame,
    features: Iterable[str],
    suffix: str = "_xs",
    winsorize_pct: float = 0.01,
    min_stocks_per_date: int = 30,
) -> pd.DataFrame:
    """Add cross-sectional z-score columns.

    For each ``feature`` and each date, compute
        (feature - cross_section_mean) / cross_section_std

    Args:
        df: Long-form DataFrame with columns ['date', 'symbol', *features].
        features: Iterable of feature column names to transform.
        suffix: Appended to each feature name for the new column.
        winsorize_pct: Clip to [pct, 1-pct] quantiles per date before
            z-scoring. Default 1% (clips top/bottom 1% on each date).
            Set to 0 to disable.
        min_stocks_per_date: Dates with fewer stocks than this get NaN
            instead of a z-score (avoid degenerate normalization).

    Returns:
        A new DataFrame with original columns + suffix columns. Rows
        ordering preserved.
    """
    df = df.copy()
    features = [f for f in features if f in df.columns]
    if not features:
        return df

    # Validate dtype: nonnumeric columns can't be z-scored, log and skip.
    bad = [f for f in features if not np.issubdtype(df[f].dtype, np.number)]
    if bad:
        print(f"[xs] Skipping non-numeric features: {bad}")
        features = [f for f in features if f not in bad]

    # Build the transform per-date for all features at once
    groupby = df.groupby("date", sort=False)
    size_per_date = groupby.size()
    too_small = set(size_per_date[size_per_date < min_stocks_per_date].index)

    if too_small:
        print(f"[xs] {len(too_small)} dates have <{min_stocks_per_date} "
              f"stocks; their xs columns will be NaN.")

    out_cols = {}
    for feat in features:
        new_col = f"{feat}{suffix}"

        def _transform(group: pd.Series) -> pd.Series:
            if group.name in too_small:
                return pd.Series(np.nan, index=group.index)
            s = group
            if winsorize_pct > 0:
                s = _winsorize_series(s, winsorize_pct)
            mu = s.mean()
            sigma = s.std(ddof=0)
            if sigma == 0 or not np.isfinite(sigma):
                return pd.Series(0.0, index=group.index)
            return (group - mu) / sigma

        z = groupby[feat].transform(_transform)
        out_cols[new_col] = z

    new_df = df.assign(**out_cols)
    print(f"[xs] Added {len(out_cols)} cross-sectional z-score columns "
          f"(winsorize={winsorize_pct})")
    return new_df


def add_cross_sectional_rank(
    df: pd.DataFrame,
    features: Iterable[str],
    suffix: str = "_xrank",
    min_stocks_per_date: int = 30,
) -> pd.DataFrame:
    """Add cross-sectional percentile-rank columns in [0, 1].

    Equivalent to z-score for monotonic-rank purposes but robust to
    outliers. A stock at rank 0.95 is in the top 5% on that feature today.
    """
    df = df.copy()
    features = [f for f in features if f in df.columns]
    if not features:
        return df

    bad = [f for f in features if not np.issubdtype(df[f].dtype, np.number)]
    if bad:
        print(f"[xs] Skipping non-numeric features: {bad}")
        features = [f for f in features if f not in bad]

    groupby = df.groupby("date", sort=False)
    size_per_date = groupby.size()
    too_small = set(size_per_date[size_per_date < min_stocks_per_date].index)

    out_cols = {}
    for feat in features:
        new_col = f"{feat}{suffix}"

        def _transform(group: pd.Series) -> pd.Series:
            if group.name in too_small:
                return pd.Series(np.nan, index=group.index)
            return group.rank(method="average", pct=True)

        out_cols[new_col] = groupby[feat].transform(_transform)

    new_df = df.assign(**out_cols)
    print(f"[xs] Added {len(out_cols)} cross-sectional percentile-rank columns")
    return new_df


# ---------------------------------------------------------------------------
# Default feature lists -- the per-stock continuous features that benefit
# most from cross-sectional standardization. Used by prepare_prediction_data
# unless overridden in config.
# ---------------------------------------------------------------------------
DEFAULT_XS_FEATURES: List[str] = [
    # Trend / price-vs-moving-average
    "price_vs_ma_3d",
    "price_vs_ma_7d",
    "price_vs_ma_15d",
    "price_vs_ma_30d",
    "macd",
    "macd_hist",
    # Volume
    "volume_ratio_3d",
    "volume_ratio_5d",
    "volume_ratio_7d",
    "volume_ratio_15d",
    "volume_ratio_20d",
    "volume_ratio_30d",
    "volume_std_20d",
    "vwap_ratio",
    # Volatility
    "volatility_3d",
    "volatility_7d",
    "volatility_10d",
    "volatility_15d",
    "volatility_20d",
    "volatility_30d",
    "volatility_60d",
    "bb_width",
    # Momentum oscillators
    "rsi_7",
    "cci_20",
    "williams_r_14",
    "mfi_14",
    "trix_14",
    "dpo_20",
    # Returns (cross-sectional return ranks -- reversal / momentum signal)
    "returns_2d",
    "returns_3d",
    "returns_5d",
    "returns_7d",
    "returns_10d",
    "returns_15d",
    "returns_20d",
    "returns_30d",
    "returns_60d",
    "roc_3d",
    "roc_7d",
    "roc_15d",
    "roc_30d",
    "momentum_50d",
]
