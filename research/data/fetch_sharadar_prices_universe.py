"""Daily Sharadar SEP prices for an arbitrary ticker list (research
universes beyond the S&P members, e.g. data/midcap_membership.parquet).

Same client and storage convention as fetch_sharadar_prices.py (one call per
ticker from START, incremental parquet, symbol == Sharadar ticker), written
to a separate file so the S&P panel is untouched.

Usage:  python data/fetch_sharadar_prices_universe.py --membership data/midcap_membership.parquet --out data/sharadar_prices_midcap.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from data.vendor_keys import get_key  # noqa: E402
import fetch_sharadar_prices as fs  # noqa: E402

DATA = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2013-06-01")
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()
    key = get_key("SHARADAR_API_KEY")
    if not key:
        raise SystemExit("SHARADAR_API_KEY missing")
    out_path = Path(args.out)
    syms = sorted(pd.read_parquet(args.membership)["symbol"].astype(str).unique())
    have = set(pd.read_parquet(out_path, columns=["symbol"])["symbol"].unique()) if out_path.exists() else set()
    todo = [s for s in syms if s not in have]
    print(f"universe: {len(syms)} tickers | stored: {len(have)} | to fetch: {len(todo)}", flush=True)
    frames, failed, t0 = [], [], time.time()
    for n, sym in enumerate(todo, 1):
        try:
            df = fs._get("stocks", key, ticker=sym, **{"from": args.start})
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{n}/{len(todo)}] {sym}: {exc}", flush=True)
            failed.append(sym)
            continue
        if df.empty:
            failed.append(sym)
        else:
            df["symbol"] = sym
            frames.append(df)
        if n % 50 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {sym}: {len(df)} rows  {time.time() - t0:.0f}s", flush=True)
            if frames:
                new = pd.concat(frames, ignore_index=True)
                if out_path.exists():
                    old = pd.read_parquet(out_path)
                    new = pd.concat([old[~old["symbol"].isin(new["symbol"])], new], ignore_index=True)
                new.to_parquet(out_path, index=False)
                frames = []
        time.sleep(args.sleep)
    total = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    print(f"stored: {total['symbol'].nunique() if len(total) else 0} tickers, {len(total):,} rows; failed {len(failed)}: {failed[:20]}")


if __name__ == "__main__":
    sys.exit(main())
