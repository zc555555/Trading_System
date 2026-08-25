"""Fetch survivorship-complete daily prices from Sharadar (SEP table).

WHY: free yfinance data only contains companies that still exist, so the
price panel is missing every S&P member that was later delisted (the
"survivorship residual" in the report's limitations). Sharadar's SEP
covers 25k+ active AND delisted securities back to 1998. This fetcher
pulls the point-in-time S&P membership universe (data/sp500_membership.
parquet -- 500+ symbols incl. departed members) and stores raw rows.

Two tables are stored, unmodified apart from column lower-casing:
  data/sharadar_tickers.parquet : metadata incl. delisting status/dates
  data/sharadar_prices.parquet  : SEP rows (open/high/low/close/volume are
                                  split-adjusted; closeadj also dividend-
                                  adjusted; closeunadj raw)

Idempotent: symbols already stored are skipped unless --refresh. Nothing
here touches stocks.parquet; a separate, pre-registered experiment builds
the survivorship-complete panel and re-measures the baseline.

Auth: SHARADAR_API_KEY (env or config_keys.py). "test-api-key" returns
the free DJIA-30 sample -- use --sample to smoke-test without a
subscription.

Usage:
    python data/fetch_sharadar_prices.py --sample          # AAPL/MSFT smoke test
    python data/fetch_sharadar_prices.py                   # full PIT universe
    python data/fetch_sharadar_prices.py --refresh
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.vendor_keys import get_key  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
MEMBERSHIP = DATA_DIR / "sp500_membership.parquet"
PANEL = DATA_DIR / "stocks.parquet"
OUT_PRICES = DATA_DIR / "sharadar_prices.parquet"
OUT_TICKERS = DATA_DIR / "sharadar_tickers.parquet"
MANIFEST = DATA_DIR / "sharadar_manifest.json"

BASE = "https://api.sharadar.com/v1.0/data"
START = "2014-01-01"
BATCH = 25            # tickers per request (25 x ~3200 rows < 10k row limit? no -> paged)
PAGE = 10000
SLEEP = 0.25


def _get(table: str, key: str, **params) -> pd.DataFrame:
    """One (possibly paged) CSV query against a Sharadar table."""
    frames = []
    skip = 0
    while True:
        q = {"api_key": key, "format": "csv", "limit": PAGE, "skip": skip, **params}
        r = requests.get(f"{BASE}/{table}", params=q, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"{table} HTTP {r.status_code}: {r.text[:200]}")
        if not r.text.strip():
            break
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        skip += PAGE
        time.sleep(SLEEP)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower() for c in out.columns]
    return out


def universe() -> list[str]:
    syms = set()
    if MEMBERSHIP.exists():
        syms |= set(pd.read_parquet(MEMBERSHIP)["symbol"].astype(str))
    if PANEL.exists():
        syms |= set(pd.read_parquet(PANEL, columns=["symbol"])["symbol"].astype(str))
    # Sharadar uses '.' for share classes (BRK.B), yfinance uses '-'
    return sorted({s.replace("-", ".") for s in syms if s and not s.startswith("^")})


def main(sample: bool, refresh: bool) -> int:
    key = get_key("SHARADAR_API_KEY") or ("test-api-key" if sample else "")
    if not key:
        print("SHARADAR_API_KEY not set (env or config_keys.py). Use --sample for the free DJIA-30 smoke test.")
        return 1
    syms = ["AAPL", "MSFT"] if sample else universe()
    print(f"universe: {len(syms)} symbols | key: {'sample' if key == 'test-api-key' else 'set'}")

    have = set()
    if OUT_PRICES.exists() and not refresh:
        # data pulled with the free sample key is 5-year-limited: never let
        # it masquerade as a full fetch once a real key is present
        prior = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
        if prior.get("key_mode") == "sample" and key != "test-api-key":
            print("stored data came from the sample key -> refetching everything")
            refresh = True
        else:
            have = set(pd.read_parquet(OUT_PRICES, columns=["ticker"])["ticker"].unique())
    todo = [s for s in syms if s not in have]
    print(f"stored: {len(have)}, to fetch: {len(todo)}")

    # 1) ticker metadata (delisting status) for the whole universe, one call per batch
    meta_frames = []
    for i in range(0, len(syms), BATCH):
        chunk = syms[i:i + BATCH]
        try:
            meta_frames.append(_get("tickers", key, ticker=",".join(chunk)))
        except Exception as exc:  # noqa: BLE001
            print(f"  [tickers] batch {i // BATCH}: {exc}")
        time.sleep(SLEEP)
    meta = pd.concat([m for m in meta_frames if len(m)], ignore_index=True) if meta_frames else pd.DataFrame()
    if len(meta):
        meta.to_parquet(OUT_TICKERS, index=False)
        delist_col = next((c for c in meta.columns if "delist" in c), None)
        n_del = int(meta[delist_col].astype(str).str.upper().isin(["Y", "TRUE", "1"]).sum()) if delist_col else -1
        print(f"tickers table: {len(meta)} rows, columns {list(meta.columns)[:12]}... | delisted flagged: {n_del}")

    # 2) prices, one call per ticker (paged if > 10k rows)
    frames, missing = [], []
    for n, sym in enumerate(todo, 1):
        try:
            df = _get("stocks", key, ticker=sym, **{"from": START})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{n}/{len(todo)}] {sym}: {exc}")
            missing.append(sym)
            continue
        if df.empty:
            missing.append(sym)
        else:
            frames.append(df)
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {sym}: {len(df)} rows")
            if frames:  # incremental save
                new = pd.concat(frames, ignore_index=True)
                if OUT_PRICES.exists() and not refresh:
                    old = pd.read_parquet(OUT_PRICES)
                    new = pd.concat([old[~old["ticker"].isin(new["ticker"])], new], ignore_index=True)
                new.to_parquet(OUT_PRICES, index=False)
                frames = [] if not refresh else frames
                refresh = False
        time.sleep(SLEEP)

    total = pd.read_parquet(OUT_PRICES) if OUT_PRICES.exists() else pd.DataFrame()
    manifest = {
        "fetched_at": datetime.now().isoformat(), "start": START,
        "key_mode": "sample" if key == "test-api-key" else "subscription",
        "universe": len(syms), "stored_tickers": int(total["ticker"].nunique()) if len(total) else 0,
        "rows": int(len(total)), "missing": missing,
        "date_min": str(total["date"].min()) if len(total) else None,
        "date_max": str(total["date"].max()) if len(total) else None,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nstored tickers: {manifest['stored_tickers']}, rows: {manifest['rows']:,}, "
          f"{manifest['date_min']} .. {manifest['date_max']}")
    print(f"missing ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="free DJIA-30 smoke test with test-api-key")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.sample, a.refresh))
