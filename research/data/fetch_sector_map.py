"""Fetch a sector classification for every symbol in the price panel.

One-time fetch with a persistent cache: sectors change rarely (GICS
reclassifications are ~yearly events), so the map is committed to the repo
and only re-fetched for symbols missing from the cache. Re-running is
therefore cheap and idempotent.

Source: yfinance ``Ticker.info['sector']`` (Yahoo's taxonomy, 11 sectors,
near-GICS). ETFs and lookup failures get sector "Unknown" -- the risk
model treats Unknown as its own bucket rather than guessing.

Usage:
    python data/fetch_sector_map.py            # fill gaps only
    python data/fetch_sector_map.py --refresh  # re-fetch everything
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent
PANEL_PATH = DATA_DIR / "stocks.parquet"
OUT_PATH = DATA_DIR / "sector_map.csv"


def load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_csv(OUT_PATH)
    return pd.DataFrame(columns=["symbol", "sector", "industry"])


def fetch_one(symbol: str) -> tuple[str, str]:
    """Return (sector, industry); 'Unknown' on failure or for ETFs."""
    for attempt in range(3):
        try:
            info = yf.Ticker(symbol).info
            if info.get("quoteType") == "ETF":
                return "ETF", "ETF"
            sector = info.get("sector") or "Unknown"
            industry = info.get("industry") or "Unknown"
            return sector, industry
        except Exception as exc:  # noqa: BLE001 -- throttling, delisted, etc.
            wait = 2 * (attempt + 1)
            print(f"  {symbol}: attempt {attempt + 1} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    return "Unknown", "Unknown"


def main(refresh: bool = False) -> None:
    symbols = sorted(pd.read_parquet(PANEL_PATH, columns=["symbol"])["symbol"].unique())
    existing = load_existing()
    have = set() if refresh else set(existing.loc[existing["sector"] != "Unknown", "symbol"])
    todo = [s for s in symbols if s not in have]
    print(f"panel symbols: {len(symbols)}, cached: {len(have)}, to fetch: {len(todo)}")

    rows = existing[existing["symbol"].isin(have)].to_dict("records")
    for i, sym in enumerate(todo, 1):
        sector, industry = fetch_one(sym)
        rows.append({"symbol": sym, "sector": sector, "industry": industry})
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {sym}: {sector}")
            # incremental save so an interrupt loses at most 25 fetches
            pd.DataFrame(rows).sort_values("symbol").to_csv(OUT_PATH, index=False)
        time.sleep(0.4)  # stay far from Yahoo throttle

    df = pd.DataFrame(rows).sort_values("symbol")
    df.to_csv(OUT_PATH, index=False)
    print(f"\nsaved {len(df)} rows -> {OUT_PATH}")
    print(df["sector"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    main(refresh=args.refresh)
