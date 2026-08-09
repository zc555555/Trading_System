"""Point-in-time S&P 500 membership from Wikipedia (merged-B item 2).

Scrapes the current-constituents and historical-changes tables, then
reconstructs membership INTERVALS per ticker by replaying the change log
backward from today's membership set.

Output: data/sp500_membership.parquet  [symbol, start, end]
    start/end are the effective dates of index membership (end = NaT for
    current members). Symbols normalized to the price-data convention
    (dots -> dashes, e.g. BRK.B -> BRK-B).

Known residuals (documented, not silently ignored):
- Delisted members whose price history yfinance lacks cannot be added
  back to the backtest; this filter only removes not-yet-added members
  (the S&P inclusion look-ahead), which is the measurable half.
- Ticker strings reused by different companies decades apart are not
  disambiguated; impact within the 2014+ evaluation window is minor.
- Rows with a missing Added or Removed side (spin-offs etc.) contribute
  only their populated side.

Usage:
    python data/fetch_sp500_history.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = Path(__file__).parent / "sp500_membership.parquet"
FAR_PAST = pd.Timestamp("1957-03-04")   # index inception


def norm(t) -> str | None:
    if not isinstance(t, str) or not t.strip():
        return None
    return t.strip().upper().replace(".", "-")


def fetch_tables():
    html = requests.get(URL, headers={
        "User-Agent": "Mozilla/5.0 (research; stock_predict)"}, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    current = tables[0]
    changes = tables[1]
    changes.columns = ["date", "added", "added_name", "removed",
                       "removed_name", "reason", "extra"][:len(changes.columns)]
    return current, changes


def build_intervals(current: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    changes = changes.copy()
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"]).sort_values("date")

    # Per-ticker event list, oldest first: (+1 added, -1 removed)
    events: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for _, r in changes.iterrows():
        a, d = norm(r["added"]), norm(r["removed"])
        if a:
            events.setdefault(a, []).append((r["date"], +1))
        if d:
            events.setdefault(d, []).append((r["date"], -1))

    current_set = {norm(s) for s in current["Symbol"] if norm(s)}

    rows = []
    all_tickers = set(events) | current_set
    for tkr in sorted(all_tickers):
        evs = sorted(events.get(tkr, []))
        # Walk forward: an open interval starts at an 'added' (or FAR_PAST
        # if the first event is a removal / no events but currently listed)
        open_start = None
        if not evs or evs[0][1] == -1:
            open_start = FAR_PAST
        for date, kind in evs:
            if kind == +1:
                if open_start is None:
                    open_start = date
                # duplicate 'added' while open: keep the earliest start
            else:
                if open_start is not None:
                    rows.append((tkr, open_start, date))
                    open_start = None
                # removal with no open interval: ignore (already closed)
        if open_start is not None:
            if tkr in current_set:
                rows.append((tkr, open_start, pd.NaT))
            else:
                # interval left open but not currently a member: the change
                # log is missing its removal; close it at the last event to
                # stay conservative (membership only when evidence exists)
                rows.append((tkr, open_start, evs[-1][0] if evs else pd.NaT))

    out = pd.DataFrame(rows, columns=["symbol", "start", "end"])
    return out


def sanity_check(intervals: pd.DataFrame):
    print("\nmembership count on sample dates (expect ~500-505 post-2000):")
    for y in range(2014, 2027):
        d = pd.Timestamp(f"{y}-06-30")
        n = ((intervals["start"] <= d)
             & (intervals["end"].isna() | (intervals["end"] > d))).sum()
        print(f"  {d.date()}: {n}")


def main():
    current, changes = fetch_tables()
    print(f"current constituents: {len(current)}, change rows: {len(changes)}")
    intervals = build_intervals(current, changes)
    print(f"intervals: {len(intervals)} for {intervals['symbol'].nunique()} tickers")
    sanity_check(intervals)
    intervals.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
