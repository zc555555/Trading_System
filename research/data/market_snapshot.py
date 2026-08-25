"""Frozen, append-only snapshot of the macro/market series behind the
market features (SPY, ^VIX, QQQ, DIA, IWM, TLT, GLD, USO, UUP).

WHY: the feature builder used to call ``yf.Ticker(sym).history()`` live on
every rebuild.  Yahoo returns dividend-ADJUSTED closes, and every new
dividend re-adjusts the entire history, so 1/5/20-day return features
drifted between runs -- observed as a virgin-tier IC moving 0.0168 ->
0.0141 with no code change (report §6.4).  A ruler whose reading changes
between two identical measurements is not a ruler.

HOW: this module keeps RAW closes + dividends + splits per symbol/date in
``market_snapshot.csv`` (committed to git via a .gitignore exception).
Rules:
  * rows already in the snapshot are NEVER modified by a rebuild -- only
    dates after the symbol's last stored date are fetched and appended;
  * a consistent total-return index is derived deterministically from the
    frozen raw rows (see ``adjusted_close``), so features are a pure
    function of the file, not of Yahoo's current adjustment state;
  * to refresh history on purpose (a known upstream correction), delete
    the symbol's rows (or the file) and re-run -- the change then shows
    up as a reviewable git diff instead of silent drift.

The fetch function is injectable for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = DATA_DIR / "market_snapshot.csv"
COLUMNS = ["symbol", "date", "close", "dividends", "splits"]
FIRST_FETCH_START = "2010-01-01"
OVERLAP_DAYS = 10          # re-fetch window before last stored date (gap safety)
MARKET_TZ = "America/New_York"


def _yf_fetch(symbol: str, start, end) -> pd.DataFrame:
    """Raw (unadjusted) daily history -> DataFrame(date, close, dividends, splits)."""
    import yfinance as yf
    hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False,
                                     actions=True)
    if hist is None or len(hist) == 0:
        return pd.DataFrame(columns=COLUMNS[1:])
    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_convert(MARKET_TZ)
    out = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "close": hist["Close"].to_numpy(dtype=float),
        "dividends": hist["Dividends"].to_numpy(dtype=float)
        if "Dividends" in hist else 0.0,
        "splits": hist["Stock Splits"].to_numpy(dtype=float)
        if "Stock Splits" in hist else 0.0,
    })
    return out[out["close"].notna()].reset_index(drop=True)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, dtype={"symbol": str, "date": str})
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_snapshot(df: pd.DataFrame, path: Path = SNAPSHOT_PATH) -> None:
    df = df[COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def last_completed_session(now=None) -> str:
    """Latest NY calendar date whose regular session has closed (16:15 ET
    cushion for the official close print).  Freezing an intraday bar
    would poison the snapshot permanently, so nothing at or after this
    date is ever stored."""
    now = pd.Timestamp.now(tz=MARKET_TZ) if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize(MARKET_TZ)
    else:
        now = now.tz_convert(MARKET_TZ)
    cutoff = now.normalize() + pd.Timedelta(hours=16, minutes=15)
    day = now.normalize() if now >= cutoff else now.normalize() - pd.Timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def update_snapshot(symbols: list[str], end_date,
                    fetch: Callable = _yf_fetch,
                    path: Path = SNAPSHOT_PATH,
                    verbose: bool = True,
                    now=None) -> pd.DataFrame:
    """Append rows after each symbol's last stored date; never rewrite.
    Rows are only ever stored for sessions that have already closed."""
    snap = load_snapshot(path)
    end_str = min(pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                  last_completed_session(now))
    pieces = [snap] if len(snap) else []
    for sym in symbols:
        have = snap[snap["symbol"] == sym]
        if have.empty:
            start = FIRST_FETCH_START
            floor = None
        else:
            last = have["date"].max()
            floor = last
            start = (pd.Timestamp(last) - pd.Timedelta(days=OVERLAP_DAYS)
                     ).strftime("%Y-%m-%d")
        if floor is not None and floor >= end_str:
            continue
        try:
            new = fetch(sym, start, end_str)
        except Exception as exc:  # noqa: BLE001 -- keep the frozen history usable
            if verbose:
                print(f"  [snapshot] {sym}: fetch failed ({exc}); using stored rows only")
            continue
        if floor is not None:
            new = new[new["date"] > floor]          # FROZEN: never touch stored dates
        new = new[new["date"] <= end_str]
        if len(new):
            new = new.assign(symbol=sym)[COLUMNS]
            pieces.append(new)
        if verbose:
            print(f"  [snapshot] {sym}: stored {len(have)} rows, appended {len(new)}"
                  f"{' (first fetch)' if floor is None else ''}")
    snap = pd.concat(pieces, ignore_index=True) if pieces else snap
    snap = snap.drop_duplicates(["symbol", "date"], keep="first")
    save_snapshot(snap, path)
    return snap


def adjusted_close(sym_rows: pd.DataFrame) -> pd.Series:
    """Deterministic total-return index from frozen raw rows.

        r_t = (close_t * split_t + div_t) / close_{t-1} - 1
        adj_t = adj_{t-1} * (1 + r_t),  adj_0 = close_0

    split_t is Yahoo's ratio on the split date (new/old; 0 = no split).
    Multiplying the post-split close by the ratio makes it comparable with
    the pre-split previous close.  Symbols without dividends/splits (^VIX)
    reproduce the raw close exactly.
    """
    rows = sym_rows.sort_values("date")
    close = rows["close"].to_numpy(dtype=float)
    div = np.nan_to_num(rows["dividends"].to_numpy(dtype=float))
    split = np.nan_to_num(rows["splits"].to_numpy(dtype=float))
    split = np.where(split > 0, split, 1.0)
    prev = np.concatenate([[np.nan], close[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (close * split + div) / prev - 1.0
    r[0] = 0.0
    adj = close[0] * np.cumprod(1.0 + r)
    return pd.Series(adj, index=rows.index)


def market_history(symbol: str, snap: pd.DataFrame, start_date, tz) -> pd.DataFrame:
    """Feature-builder view: DataFrame(date, close) where ``close`` is the
    consistent adjusted index (raw close for symbols without actions), with
    dates localized to ``tz`` to match the stock panel."""
    rows = snap[snap["symbol"] == symbol].sort_values("date")
    if rows.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({
        "date": pd.to_datetime(rows["date"].to_numpy()),
        "close": adjusted_close(rows).to_numpy(),
        "raw_close": rows["close"].to_numpy(dtype=float),
    })
    if tz is not None:
        out["date"] = out["date"].dt.tz_localize(tz)
    start = pd.Timestamp(start_date)
    if start.tzinfo is None and tz is not None:
        start = start.tz_localize(tz)
    elif start.tzinfo is not None and tz is None:
        start = start.tz_localize(None)
    return out[out["date"] >= start].reset_index(drop=True)
