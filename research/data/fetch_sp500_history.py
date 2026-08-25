"""Point-in-time S&P 500 membership from Wikipedia (merged-B item 2).

Scrapes the current-constituents table and the historical-changes table,
then reconstructs membership INTERVALS per ticker by replaying the change
log against today's membership set.

Output: data/sp500_membership.parquet  [symbol, start, end]
    start/end are the effective dates of index membership (end = NaT for
    current members). Symbols normalized to the price-data convention
    (dots -> dashes, e.g. BRK.B -> BRK-B).

2026-08-25 incident: Wikipedia moved the change log from the main list
page to a separate article ("Historical components of the S&P 500").
The old parser silently read a navbox as the change log (11 rows), so the
2026-08-16 weekly refresh overwrote the table with a DEGENERATE one --
503 current members, every start = 1957, zero removals -- and the
inclusion-look-ahead filter quietly stopped filtering. Two fixes:
  * the change log is read from the new article (multi-level header
    flattened), and the constituents table's own "Date added" column
    seeds start dates for current members that the log does not cover;
  * a HARD GUARD refuses to write the parquet unless the change log has
    >= MIN_CHANGE_ROWS rows and membership counts actually vary across
    years. A failed guard exits non-zero so the scheduled wrapper logs it.

Known residuals (documented, not silently ignored):
- Ticker strings reused by different companies decades apart are not
  disambiguated; impact within the 2014+ evaluation window is minor.
- Rows with a missing Added or Removed side (spin-offs etc.) contribute
  only their populated side.
- Wikipedia's log is "selected" changes, not exhaustive; departed members
  it omits stay invisible. Cross-check candidate: github.com/fja05680/
  sp500 (monthly constituent lists since 1996).

Usage:
    python data/fetch_sp500_history.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

URL_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_CHANGES = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
OUT = Path(__file__).parent / "sp500_membership.parquet"
FAR_PAST = pd.Timestamp("1957-03-04")   # index inception
MIN_CHANGE_ROWS = 200
HEADERS = {"User-Agent": "Mozilla/5.0 (research; stock_predict)"}


def norm(t) -> str | None:
    if not isinstance(t, str) or not t.strip():
        return None
    t = t.strip().upper().replace(".", "-")
    return t if t.replace("-", "").isalnum() else None


def _flatten(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if str(p) != "nan"]
            out.append(" ".join(dict.fromkeys(parts)).strip().lower())
        else:
            out.append(str(c).strip().lower())
    return out


def _find_changes_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """The change log is the table whose flattened header mentions both
    'added' and 'removed' (robust to column reordering / renaming)."""
    for t in tables:
        cols = _flatten(t.columns)
        joined = " ".join(cols)
        if "added" in joined and "removed" in joined and len(t) >= 50:
            t = t.copy()
            t.columns = cols
            return t
    return None


def fetch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    html = requests.get(URL_CURRENT, headers=HEADERS, timeout=30).text
    current = pd.read_html(io.StringIO(html))[0]
    if "Symbol" not in current.columns:
        raise RuntimeError("constituents table not recognised (no 'Symbol' column)")

    html = requests.get(URL_CHANGES, headers=HEADERS, timeout=30).text
    changes = _find_changes_table(pd.read_html(io.StringIO(html)))
    if changes is None:
        raise RuntimeError(f"change log table not found at {URL_CHANGES}")
    date_col = next(c for c in changes.columns if "date" in c)
    add_col = next(c for c in changes.columns if "added" in c and "ticker" in c)
    rem_col = next(c for c in changes.columns if "removed" in c and "ticker" in c)
    changes = changes.rename(columns={date_col: "date", add_col: "added", rem_col: "removed"})
    return current, changes[["date", "added", "removed"]]


def build_intervals(current: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    changes = changes.copy()
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"]).sort_values("date")

    events: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for _, r in changes.iterrows():
        a, d = norm(r["added"]), norm(r["removed"])
        if a:
            events.setdefault(a, []).append((r["date"], +1))
        if d:
            events.setdefault(d, []).append((r["date"], -1))

    current_set = {norm(s) for s in current["Symbol"] if norm(s)}
    # constituents table carries its own inclusion date -- second source
    date_added = {}
    if "Date added" in current.columns:
        for s, d in zip(current["Symbol"], pd.to_datetime(current["Date added"], errors="coerce")):
            if norm(s) and pd.notna(d):
                date_added[norm(s)] = d

    rows = []
    for tkr in sorted(set(events) | current_set):
        evs = sorted(events.get(tkr, []))
        open_start = None
        if not evs or evs[0][1] == -1:
            # no add event in the log: use the constituents table's date
            # for current members, else index inception
            open_start = date_added.get(tkr, FAR_PAST)
        for date, kind in evs:
            if kind == +1:
                if open_start is None:
                    open_start = date
            else:
                if open_start is not None:
                    rows.append((tkr, open_start, date))
                    open_start = None
        if open_start is not None:
            if tkr in current_set:
                rows.append((tkr, open_start, pd.NaT))
            else:
                rows.append((tkr, open_start, evs[-1][0] if evs else pd.NaT))
    return pd.DataFrame(rows, columns=["symbol", "start", "end"])


def membership_counts(intervals: pd.DataFrame) -> dict:
    out = {}
    for y in range(2014, 2027):
        d = pd.Timestamp(f"{y}-06-30")
        out[y] = int(((intervals["start"] <= d)
                      & (intervals["end"].isna() | (intervals["end"] > d))).sum())
    return out


def guard(changes: pd.DataFrame, intervals: pd.DataFrame, counts: dict) -> None:
    """Refuse to overwrite a good table with a degenerate one."""
    if len(changes) < MIN_CHANGE_ROWS:
        raise RuntimeError(f"GUARD: change log has {len(changes)} rows (< {MIN_CHANGE_ROWS}) -- "
                           f"parser is reading the wrong table; parquet NOT written")
    if intervals["end"].notna().sum() == 0:
        raise RuntimeError("GUARD: zero closed intervals (no removals parsed); parquet NOT written")
    if len(set(counts.values())) == 1:
        raise RuntimeError("GUARD: membership count identical on every year; parquet NOT written")


def main() -> int:
    current, changes = fetch_tables()
    print(f"current constituents: {len(current)}, change rows: {len(changes)}")
    intervals = build_intervals(current, changes)
    counts = membership_counts(intervals)
    print(f"intervals: {len(intervals)} for {intervals['symbol'].nunique()} tickers; "
          f"closed (departed) intervals: {int(intervals['end'].notna().sum())}")
    print("membership count on sample dates (expect ~500-505 post-2000):")
    for y, n in counts.items():
        print(f"  {y}-06-30: {n}")
    guard(changes, intervals, counts)
    intervals.to_parquet(OUT, index=False)
    print(f"\nsaved: {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- unattended: loud, non-zero, no overwrite
        print(f"[FAIL] sp500 membership refresh: {exc}")
        sys.exit(1)
