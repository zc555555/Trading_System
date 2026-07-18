"""Per-symbol ATR_14 computed from the research OHLC parquet.

ATR (Average True Range, Wilder 1978) = rolling mean of the true range
where true_range = max(high - low, |high - prev_close|, |low - prev_close|).

The build_dataset feature pipeline doesn't currently emit atr_14, so we
compute it here on demand from raw OHLC. Cheap (one parquet read, vectorized
groupby) and avoids forcing a feature-pipeline rebuild just for stops.

If the parquet is missing or a symbol has too little history, the loader
returns None and the caller falls back to a fixed-percentage stop.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PARQUET = (
    Path(__file__).resolve().parent.parent
    / "research" / "data" / "stocks_with_time_windows.parquet"
)
ATR_WINDOW = 14
MIN_HISTORY = 20  # need at least one full window plus a few extras


@functools.lru_cache(maxsize=1)
def _load_latest_atr_table() -> pd.DataFrame:
    """Compute and cache the latest ATR_14 + close for every symbol.

    Returns a DataFrame indexed by row position with columns
    ['symbol', 'atr_14', 'close']. One row per symbol.
    """
    if not _PARQUET.exists():
        return pd.DataFrame(columns=["symbol", "atr_14", "close"])

    df = pd.read_parquet(
        _PARQUET, columns=["date", "symbol", "high", "low", "close"]
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Previous close per symbol (NaN for the first row of each symbol)
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)

    # True range = max of three candidates
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - df["prev_close"]).abs()
    low_prev = (df["low"] - df["prev_close"]).abs()
    df["tr"] = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)

    # Wilder's ATR is technically an EMA of TR but the SMA approximation is fine
    # for risk sizing. We use rolling mean for clarity.
    df["atr_14"] = (
        df.groupby("symbol")["tr"]
        .transform(lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean())
    )

    # Latest row per symbol with a valid ATR
    df = df.dropna(subset=["atr_14"])
    # Filter out symbols with too little history (need at least MIN_HISTORY rows)
    counts = df.groupby("symbol").size()
    keep = counts[counts >= MIN_HISTORY].index
    df = df[df["symbol"].isin(keep)]

    latest = df.groupby("symbol", as_index=False).tail(1)
    return latest[["symbol", "atr_14", "close"]].reset_index(drop=True)


def get_atr(symbol: str) -> Optional[float]:
    """Return the most recent ATR_14 (in dollars) for ``symbol``, or None."""
    tbl = _load_latest_atr_table()
    if tbl.empty:
        return None
    hit = tbl.loc[tbl["symbol"] == symbol, "atr_14"]
    if hit.empty:
        return None
    val = float(hit.iloc[0])
    return val if val > 0 and np.isfinite(val) else None


def atr_summary() -> str:
    """One-liner status string for orchestrator logs."""
    tbl = _load_latest_atr_table()
    if tbl.empty:
        return f"ATR table EMPTY (parquet missing at {_PARQUET})"
    return f"ATR_14 computed for {len(tbl)} symbols from {_PARQUET.name}"


if __name__ == "__main__":
    print(atr_summary())
    print()
    print(f"{'symbol':<10}{'close':>10}{'ATR_14':>10}{'ATR%':>8}"
          f"{'2xATR$':>10}{'2xATR%':>10}")
    print("-" * 60)
    for s in ("AAPL", "NVDA", "TSLA", "KO", "MSFT", "WMT", "JPM", "PFE",
              "CRWD", "PANW", "MU", "INTC"):
        atr = get_atr(s)
        tbl = _load_latest_atr_table()
        cur_hit = tbl.loc[tbl["symbol"] == s, "close"]
        if atr is None or cur_hit.empty:
            print(f"  {s:<8}  missing")
            continue
        cur = float(cur_hit.iloc[0])
        pct = atr / cur * 100
        print(f"  {s:<8}{cur:>10.2f}{atr:>10.3f}{pct:>7.2f}%"
              f"{2*atr:>10.2f}{2*atr/cur*100:>9.2f}%")
