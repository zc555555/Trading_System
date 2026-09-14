"""The production selection policy as ONE module, shared by the nightly
signal script and the evaluation replay (2026-09-14 review item 3).

Everything the live book applies between "per-date factor scores" and
"target positions" lives here, each step behind a switch in
SelectionParams, so evaluation can replay exactly what production does and
each step can be tested as a pre-registered hypothesis:

  1. smoothing     the last `smooth_days` sessions' per-date z-scored factor
                   scores are averaged with `day_weights` (newest first);
                   a symbol with fewer sessions gets the LAST k weights,
                   renormalised, newest still first (the production rule);
  2. blending      factors combine with the training run's effective weights;
  3. trend filter  7d / 14d return and MA7-vs-MA30 strength thresholds
                   (config_trend_filters), applied to longs AND shorts;
                   a symbol with fewer than 30 sessions fails it;
  4. selection     top_n by |prediction|, min_confidence, min_stocks;
  5. sizing        confidence / clip(volatility_20d, vol_floor), normalised.

News sentiment is NOT part of selection any more (decision 2026-09-14): the
signal script scores it and records it as a diagnostic column only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DAY_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)     # newest session first
VERSION = "selection-2026-09-14"
TREND_LOOKBACK = 30


@dataclass(frozen=True)
class SelectionParams:
    smooth_days: int = 5
    day_weights: tuple = DAY_WEIGHTS
    trend_filter: bool = True
    filter_7d_min: Optional[float] = -3.0
    filter_14d_min: Optional[float] = -6.0
    filter_30d_min: Optional[float] = None
    trend_strength_min: Optional[float] = -3.0
    require_above_mas: bool = False
    top_n: int = 10
    min_confidence: float = 0.0015          # get_daily_signals config['min_confidence']
    min_stocks: int = 3
    inv_vol: bool = True
    vol_floor: float = 0.006
    version: str = VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["day_weights"] = list(self.day_weights)
        return d

    @classmethod
    def production(cls, **overrides) -> "SelectionParams":
        """The live policy: trend thresholds from config_trend_filters, the
        rest from the defaults above (which ARE the production values)."""
        base: dict = {}
        try:
            import config_trend_filters as tf
            base = dict(filter_7d_min=tf.FILTER_7D_MIN_RETURN, filter_14d_min=tf.FILTER_14D_MIN_RETURN,
                        filter_30d_min=tf.FILTER_30D_MIN_RETURN, trend_strength_min=tf.FILTER_TREND_STRENGTH_MIN,
                        require_above_mas=bool(tf.REQUIRE_PRICE_ABOVE_MAs))
        except ImportError:
            pass
        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------
# 1. smoothing
# ---------------------------------------------------------------------------
def _weights_for(params: SelectionParams) -> np.ndarray:
    w = np.asarray(params.day_weights[:params.smooth_days], dtype=float)
    return w / w.sum()


def _tail_sums(w: np.ndarray) -> np.ndarray:
    """tail[k] = sum of the last k weights (k = 0..m); tail[0] = 1 (unused)."""
    m = len(w)
    return np.array([1.0] + [w[m - k:].sum() for k in range(1, m + 1)])


def smooth_scores(scores: pd.DataFrame, value_cols: Sequence[str], params: SelectionParams) -> pd.DataFrame:
    """One row per symbol: the day-weighted mean of `value_cols` over the
    LAST `smooth_days` sessions of `scores` ([date, symbol, ...]).

    Production rule, reproduced exactly: with k sessions available for a
    symbol, the last k weights are renormalised and assigned newest-first.
    NaN values propagate (a missing score is not silently skipped)."""
    value_cols = list(value_cols)
    if params.smooth_days <= 1:
        last = scores.sort_values("date").groupby("symbol", sort=False).tail(1)
        return last[["symbol", *value_cols]].reset_index(drop=True)
    sessions = np.sort(scores["date"].unique())[-params.smooth_days:]
    s = scores[scores["date"].isin(sessions)].sort_values(["symbol", "date"], kind="mergesort")
    if s.empty:
        return pd.DataFrame(columns=["symbol", *value_cols])
    w = _weights_for(params)
    m = len(w)
    tail = _tail_sums(w)
    k = s.groupby("symbol", sort=False)["date"].transform("size").to_numpy()
    r = s.groupby("symbol", sort=False).cumcount(ascending=False).to_numpy()     # 0 = newest
    cell_w = w[m - k + r] / tail[k]
    weighted = s[value_cols].to_numpy(dtype=float) * cell_w[:, None]
    out = pd.DataFrame(weighted, columns=value_cols, index=s.index)
    out["symbol"] = s["symbol"].to_numpy()
    agg = out.groupby("symbol", sort=True)[value_cols].sum(min_count=1)
    # NaN propagates: a symbol with any NaN score in the window has no prediction
    has_nan = out.groupby("symbol", sort=True)[value_cols].apply(lambda g: g.isna().any())
    agg = agg.mask(has_nan)
    return agg.reset_index()


def smooth_panel(values: pd.DataFrame, value_col: str, params: SelectionParams) -> pd.Series:
    """Smoothing for EVERY session of a long panel ([date, symbol, value_col]),
    session-indexed: the window for date t is the last `smooth_days`
    sessions of the panel ending at t, and a symbol's k available values
    inside it get the last k weights newest-first (same rule as
    smooth_scores). Returns a Series aligned to `values.index`."""
    if params.smooth_days <= 1:
        return values[value_col].astype(float)
    wide = values.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="first", dropna=False)
    wide = wide.sort_index()
    w = _weights_for(params)
    m = len(w)
    tail = _tail_sums(w)
    lags = np.stack([wide.shift(L).to_numpy(dtype=float) for L in range(m)])   # (m, T, S); lag 0 = newest
    avail = ~np.isnan(lags)
    k = avail.sum(axis=0)                                                       # (T, S)
    j = np.cumsum(avail, axis=0) - 1                                            # recency order among available
    idx = np.clip(m - k[None, :, :] + j, 0, m - 1)
    cell_w = np.where(avail, w[idx] / tail[np.clip(k, 0, m)][None, :, :], 0.0)
    sm = np.where(k > 0, np.nansum(lags * cell_w, axis=0), np.nan)
    out = pd.DataFrame(sm, index=wide.index, columns=wide.columns).stack(dropna=False)
    out.name = value_col
    key = pd.MultiIndex.from_arrays([values["date"], values["symbol"]])
    return pd.Series(out.reindex(key).to_numpy(), index=values.index, name=value_col)


# ---------------------------------------------------------------------------
# 2. blending
# ---------------------------------------------------------------------------
def blend(frame: pd.DataFrame, factor_weights: dict, prefix: str = "factor_") -> pd.Series:
    """prediction = sum of factor score x effective weight over the factors
    present in both the frame and the weights."""
    total = pd.Series(0.0, index=frame.index)
    for c in frame.columns:
        if c.startswith(prefix) and c[len(prefix):] in factor_weights:
            total = total + frame[c].astype(float) * float(factor_weights[c[len(prefix):]])
    return total


# ---------------------------------------------------------------------------
# 3. trend filter
# ---------------------------------------------------------------------------
TREND_COLS = ["return_7d", "return_14d", "return_30d", "ma_7", "ma_14", "ma_30", "trend_strength",
              "above_ma7", "above_ma14", "above_ma30"]


def trend_metrics_panel(prices: pd.DataFrame, lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    """Per (date, symbol) the production trend metrics, computed over the
    symbol's own last `lookback` rows ending at that date (NaN until it has
    that many). Mirrors the signal script's per-symbol loop exactly:
    return_7d uses the close 6 rows back (iloc[-7] of the last 30),
    return_14d 13 rows back, return_30d the first of the 30."""
    px = prices[["date", "symbol", "close"]].sort_values(["symbol", "date"], kind="mergesort").copy()
    g = px.groupby("symbol", sort=False)["close"]
    cur = px["close"].astype(float)
    p7, p14, p30 = g.shift(6), g.shift(13), g.shift(lookback - 1)
    ma7 = g.transform(lambda s: s.rolling(7).mean())
    ma14 = g.transform(lambda s: s.rolling(14).mean())
    ma30 = g.transform(lambda s: s.rolling(lookback).mean())
    out = px[["date", "symbol"]].copy()
    out["return_7d"] = (cur - p7) / p7 * 100
    out["return_14d"] = (cur - p14) / p14 * 100
    out["return_30d"] = (cur - p30) / p30 * 100
    out["ma_7"], out["ma_14"], out["ma_30"] = ma7, ma14, ma30
    out["trend_strength"] = (ma7 - ma30) / ma30 * 100
    out["above_ma7"] = cur > ma7
    out["above_ma14"] = cur > ma14
    out["above_ma30"] = cur > ma30
    # a symbol with fewer than `lookback` rows has no metrics at all (production skips it)
    short = ma30.isna()
    out.loc[short, ["return_7d", "return_14d", "return_30d", "ma_7", "ma_14", "trend_strength"]] = np.nan
    out.loc[short, ["above_ma7", "above_ma14", "above_ma30"]] = False
    return out


def trend_metrics(prices: pd.DataFrame, as_of=None, lookback: int = TREND_LOOKBACK) -> pd.DataFrame:
    """One row per symbol as of the last date (or `as_of`)."""
    px = prices if as_of is None else prices[prices["date"] <= as_of]
    panel = trend_metrics_panel(px, lookback=lookback)
    last = panel.groupby("symbol", sort=False).tail(1)
    return last.drop(columns=["date"]).reset_index(drop=True)


def trend_mask(frame: pd.DataFrame, params: SelectionParams) -> pd.Series:
    """True where the row passes the trend filter (NaN metrics fail, as a
    NaN comparison does in production)."""
    if not params.trend_filter:
        return pd.Series(True, index=frame.index)
    ok = pd.Series(True, index=frame.index)
    if params.filter_7d_min is not None:
        ok &= frame["return_7d"] > params.filter_7d_min
    if params.filter_14d_min is not None:
        ok &= frame["return_14d"] > params.filter_14d_min
    if params.trend_strength_min is not None:
        ok &= frame["trend_strength"] > params.trend_strength_min
    if params.filter_30d_min is not None:
        ok &= frame["return_30d"] > params.filter_30d_min
    if params.require_above_mas:
        ok &= frame["above_ma7"].fillna(False).astype(bool) & frame["above_ma14"].fillna(False).astype(bool)
    return ok.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# 4 + 5. selection and sizing
# ---------------------------------------------------------------------------
def select_book(cands: pd.DataFrame, params: SelectionParams) -> pd.DataFrame:
    """`cands`: [symbol, prediction, (volatility_20d)] already trend-filtered.
    Returns the target book ordered by rank: symbol, side, prediction,
    confidence, sizing_weight, position_pct, rank -- or an EMPTY frame when
    fewer than `min_stocks` qualify (no trade)."""
    c = cands.dropna(subset=["prediction"]).copy()
    c["confidence"] = c["prediction"].astype(float).abs()
    c = c.sort_values("confidence", ascending=False, kind="mergesort").head(params.top_n)
    c = c[(c["confidence"] >= params.min_confidence) & (c["prediction"] != 0)]
    if len(c) < params.min_stocks:
        return c.iloc[0:0].assign(side=pd.Series(dtype=str), sizing_weight=pd.Series(dtype=float),
                                  position_pct=pd.Series(dtype=float), rank=pd.Series(dtype=int))
    if params.inv_vol and "volatility_20d" in c.columns:
        vol = c["volatility_20d"].astype(float).clip(lower=params.vol_floor)
        fill = vol.median() if vol.notna().any() else 0.02
        c["sizing_weight"] = c["confidence"] / vol.fillna(fill)
    else:
        c["sizing_weight"] = c["confidence"]
    c["position_pct"] = c["sizing_weight"] / c["sizing_weight"].sum() * 100.0
    c["side"] = np.where(c["prediction"] > 0, "long", "short")
    c["rank"] = np.arange(1, len(c) + 1)
    return c.reset_index(drop=True)


def daily_book(scores: pd.DataFrame, prices: pd.DataFrame, factor_weights: dict, params: SelectionParams,
               extra: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The whole policy for one as-of date (the last date of `scores`).

    scores : [date, symbol, factor_<name>...] per-date z-scored factor
             scores for at least the last `smooth_days` sessions
    prices : [date, symbol, close] with >= 30 sessions per symbol
    extra  : optional per-symbol columns joined before selection
             (volatility_20d for sizing, anything else for the record)
    Returns (book, candidates): the target book and every symbol with its
    prediction, trend metrics and `trend_ok` flag (the audit trail)."""
    factor_cols = [c for c in scores.columns if c.startswith("factor_")]
    sm = smooth_scores(scores, factor_cols, params)
    sm["prediction"] = blend(sm, factor_weights)
    metrics = trend_metrics(prices, as_of=scores["date"].max())
    cands = sm.merge(metrics, on="symbol", how="left")
    if extra is not None:
        cands = cands.merge(extra, on="symbol", how="left")
    cands["trend_ok"] = trend_mask(cands, params)
    book = select_book(cands[cands["trend_ok"]], params)
    return book, cands
