"""Fetch EDGAR companyfacts for S&P members missing from data/edgar_cache/.

    python data/fetch_edgar_missing.py [--dry-run]

fetch_fundamentals.py maps tickers to CIKs through SEC's company_tickers.json,
which lists CURRENT tickers only, so delisted members (and a few renamed
ones) never got a cache file. Sharadar's TICKERS table carries each
ticker's EDGAR link (`secfilings` ... CIK=##########), including delisted
names; this script uses it to fill the gap through the same cached
fetch_facts() call. Rerun data/build_edgar_fields.py afterwards.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import fetch_fundamentals as ff        # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
MEMBERSHIP = DATA_DIR / "sp500_membership.parquet"
PRICES = DATA_DIR / "sharadar_prices.parquet"
TICKERS = DATA_DIR / "sharadar_tickers.parquet"
CACHE = DATA_DIR / "edgar_cache"
SLEEP = 0.15                                      # SEC asks for <= 10 requests / second


def missing_with_cik() -> pd.DataFrame:
    cached = {os.path.basename(p)[:-5] for p in glob.glob(str(CACHE / "*.json"))}
    members = set(pd.read_parquet(MEMBERSHIP)["symbol"].astype(str))
    need = sorted(members - cached)
    prices = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
    tick = pd.read_parquet(TICKERS)[["ticker", "secfilings"]]
    tick["cik"] = tick["secfilings"].astype(str).str.extract(r"CIK=(\d+)", expand=False)
    m = prices[prices["symbol"].isin(need)].merge(tick, on="ticker", how="left")
    m = m[m["cik"].notna()].drop_duplicates("symbol")
    m["cik"] = m["cik"].astype(int)
    return m[["symbol", "ticker", "cik"]].reset_index(drop=True)


def main(dry_run: bool) -> int:
    m = missing_with_cik()
    print(f"members missing from the EDGAR cache with a CIK from Sharadar: {len(m)}")
    if dry_run:
        print(m.head(20).to_string(index=False))
        return 0
    ok, empty, failed = 0, 0, []
    for i, r in m.iterrows():
        try:
            facts = ff.fetch_facts(r["symbol"], int(r["cik"]))
            if facts and facts.get("facts"):
                ok += 1
            else:
                empty += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((r["symbol"], str(exc)[:100]))
        time.sleep(SLEEP)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(m)}  ok {ok}  empty {empty}  failed {len(failed)}")
    print(f"done: ok {ok}, empty {empty}, failed {len(failed)} {failed[:5]}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    sys.exit(main(ap.parse_args().dry_run))
