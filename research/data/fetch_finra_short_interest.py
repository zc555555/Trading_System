"""FINRA consolidated short interest for every PIT S&P 500 member.

Source: FINRA Query API, dataset otcMarket/consolidatedShortInterest
(anonymous access, 5,000 rows per page). Semi-monthly settlement dates
(mid-month and month-end) from 2017-12-29 onward; the two exchanges' figures
are already consolidated per symbol.

One POST per symbol (symbolCode EQUAL filter) returns its whole history
(~210 rows); renamed companies are also queried under their Sharadar
ticker (FB -> META style aliases come from sharadar_prices.parquet).

Output research/data/finra_short_interest.parquet:
    symbol            our panel symbol
    settlement_date   FINRA settlement date (position as of this date)
    short_interest    shares short
    avg_daily_volume  FINRA's average daily volume over the period
    days_to_cover     short_interest / avg_daily_volume (FINRA's field)

Causality is handled downstream (mining/aux_fields.attach_short_interest):
FINRA disseminates about seven business days after settlement, and the
field becomes usable only SHORT_INTEREST_LAG sessions after settlement.

Usage:  python data/fetch_finra_short_interest.py [--symbols A,AA] [--sleep 0.4]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent
OUT = DATA / "finra_short_interest.parquet"
PRICES = DATA / "sharadar_prices.parquet"
URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FIELDS = ["symbolCode", "settlementDate", "currentShortPositionQuantity",
          "averageDailyVolumeQuantity", "daysToCoverQuantity"]
PAGE = 5000


def symbol_pairs() -> list[tuple[str, str]]:
    """(panel symbol, FINRA ticker to query) pairs; a symbol is queried under
    its own name and, when different, its Sharadar ticker."""
    pairs = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
    out = set()
    for s, t in pairs.itertuples(index=False):
        out.add((s, s))
        if isinstance(t, str) and t and t != s:
            out.add((s, t))
    return sorted(out)


def fetch_symbol(ticker: str, session: requests.Session) -> pd.DataFrame:
    frames = []
    offset = 0
    while True:
        body = {"limit": PAGE, "offset": offset, "fields": FIELDS,
                "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": ticker}]}
        r = None
        for attempt in range(4):
            r = session.post(URL, data=json.dumps(body),
                             headers={"Content-Type": "application/json", "Accept": "text/plain"}, timeout=60)
            if r.status_code in (200, 204):          # 204 = no rows for this ticker
                break
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"{ticker}: HTTP {getattr(r, 'status_code', None)} {getattr(r, 'text', '')[:200]}")
        if r.status_code == 204 or not r.text.strip():
            break
        df = pd.read_csv(io.StringIO(r.text))
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
    if not frames:
        return pd.DataFrame(columns=FIELDS)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="comma-separated panel symbols (default: every PIT member)")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()
    pairs = symbol_pairs()
    if args.symbols:
        want = set(args.symbols.split(","))
        pairs = [p for p in pairs if p[0] in want]
    sess = requests.Session()
    frames, empty = [], []
    t0 = time.time()
    for i, (sym, tick) in enumerate(pairs, 1):
        df = fetch_symbol(tick, sess)
        if df.empty:
            empty.append(tick)
        else:
            df["symbol"] = sym
            frames.append(df)
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} tickers, {sum(len(f) for f in frames):,} rows, {time.time() - t0:.0f}s", flush=True)
        time.sleep(args.sleep)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"settlementDate": "settlement_date",
                              "currentShortPositionQuantity": "short_interest",
                              "averageDailyVolumeQuantity": "avg_daily_volume",
                              "daysToCoverQuantity": "days_to_cover"})
    out["settlement_date"] = pd.to_datetime(out["settlement_date"])
    for c in ("short_interest", "avg_daily_volume", "days_to_cover"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = (out[["symbol", "settlement_date", "short_interest", "avg_daily_volume", "days_to_cover"]]
           .sort_values(["symbol", "settlement_date"])
           .drop_duplicates(["symbol", "settlement_date"], keep="last")
           .reset_index(drop=True))
    out.to_parquet(OUT, index=False)
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} rows "
          f"({out.settlement_date.min().date()} .. {out.settlement_date.max().date()}) -> {OUT.name}")
    print(f"tickers with no FINRA rows: {len(empty)}: {empty[:40]}")


if __name__ == "__main__":
    sys.exit(main())
