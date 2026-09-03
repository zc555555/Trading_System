"""Auxiliary DSL fields beyond OHLCV, attached to a (date, symbol) panel.

Source: research/data/edgar_fields.parquet (data/build_edgar_fields.py), one
row per SEC filing with the cover-page share count -- point-in-time by
construction, because `filed` is the day the market could first know it.
A filing on session a becomes usable at session a + 1 (filings arrive
during the day; the same "second session at-or-after" convention the PEAD
experiment used for announcements).

    shares_out   latest usable cover-page share count (as-of join on filed)
    marketcap    close * shares_out / 1e6, USD millions
    turnover     volume / shares_out: fraction of shares traded in the session
    filing_days  sessions since the last usable 10-Q/10-K filing; 0 on the
                 first usable session; NaN before a symbol's first filing

Only past filings of the same symbol and same-day price/volume are read, so
the fields are causal like the rest of the DSL; the tests add a future
filing and check that past values do not move. A stale share count is
carried at most SHARES_MAX_AGE sessions (a company that stops filing drops
out rather than being priced on old numbers).

If the source file is missing the columns are absent and any expression
using them is refused by the DSL; nothing is silently zero-filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
EDGAR = DATA / "edgar_fields.parquet"
AUX_FIELDS = ("marketcap", "turnover", "filing_days")
SHARES_MAX_AGE = 130          # ~two quarters of sessions


def _naive_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s


def attach_filings(df: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """As-of join of the filing calendar onto the panel by session ordinal:
    the last filing whose usable session (filed session + 1) <= this row's
    session gives shares_out and filing_days."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ord_of = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_ord"] = ord_of.reindex(naive.to_numpy()).to_numpy()

    fl = filings[["symbol", "filed", "shares_out"]].copy()
    fl["filed"] = pd.to_datetime(fl["filed"]).dt.normalize()
    a = np.searchsorted(sessions, fl["filed"].to_numpy(dtype="datetime64[ns]"), side="left")
    keep = a < len(sessions)
    fl = fl[keep].copy()
    fl["eff_ord"] = a[keep] + 1                      # usable from the session after the filing
    fl = fl.sort_values(["eff_ord"]).drop_duplicates(["symbol", "eff_ord"], keep="last")

    left = out[["symbol", "_ord"]].reset_index().sort_values("_ord")
    merged = pd.merge_asof(left, fl[["symbol", "eff_ord", "shares_out"]].sort_values("eff_ord"),
                           left_on="_ord", right_on="eff_ord", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    age = (merged["_ord"] - merged["eff_ord"]).astype(float)
    # beyond SHARES_MAX_AGE the company has stopped filing (or the cache
    # ends): both the share count and the calendar become unknown, not stale
    fresh = age <= SHARES_MAX_AGE
    out["shares_out"] = merged["shares_out"].astype(float).where(fresh).reindex(out.index).to_numpy()
    out["filing_days"] = age.where(fresh).reindex(out.index).to_numpy()
    out["marketcap"] = out["close"].astype(float) * out["shares_out"] / 1e6
    out["turnover"] = out["volume"].astype(float) / out["shares_out"]
    for c in ("marketcap", "turnover"):
        out.loc[~np.isfinite(out[c]), c] = np.nan
    return out.drop(columns=["_ord", "shares_out"])


def attach(df: pd.DataFrame, source: Path = EDGAR, filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the auxiliary fields when the source is available; otherwise
    return df unchanged (the DSL then refuses those fields)."""
    if filings is None:
        if not Path(source).exists():
            return df
        filings = pd.read_parquet(source, columns=["symbol", "filed", "shares_out"])
    return attach_filings(df, filings)


def available(df: pd.DataFrame) -> list[str]:
    return [f for f in AUX_FIELDS if f in df.columns]
