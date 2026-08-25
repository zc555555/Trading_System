"""Fetch survivorship-complete daily prices from Sharadar (SEP table).

WHY: free yfinance data only contains companies that still exist, so the
price panel is missing every S&P member that was later delisted (the
"survivorship residual" in the report's limitations). Sharadar's SEP
covers 25k+ active AND delisted securities back to 1998. This fetcher
pulls the point-in-time S&P membership universe (data/sp500_membership.
parquet -- 500+ symbols incl. departed members) and stores raw rows.

TICKER RESOLUTION (the part that makes "delisted coverage" real):
Sharadar keeps a company under its LAST ticker. A bankrupt S&P member
lives on as its OTC code (SIVB -> SIVBQ, FRC -> FRCB) and the S&P-era
code appears only in the TICKERS table's ``relatedtickers`` field. So a
membership symbol that gets no direct hit is resolved through the full
delisted TICKERS list (relatedtickers -> current ticker) before it is
declared missing. Stored rows carry both: ``symbol`` (our membership /
yfinance-style code) and ``ticker`` (Sharadar's).

Two tables are stored, unmodified apart from column lower-casing:
  data/sharadar_tickers.parquet : metadata incl. delisting status/dates
  data/sharadar_prices.parquet  : SEP rows (open/high/low/close/volume are
                                  split-adjusted; closeadj also dividend-
                                  adjusted; closeunadj raw) + symbol

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
BATCH = 25
PAGE = 10000
SLEEP = 0.25


def _get(endpoint: str, key: str, **params) -> pd.DataFrame:
    """One (possibly paged) CSV query against a Sharadar table."""
    frames = []
    skip = 0
    while True:
        q = {"api_key": key, "format": "csv", "limit": PAGE, "skip": skip, **params}
        r = requests.get(f"{BASE}/{endpoint}", params=q, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"{endpoint} HTTP {r.status_code}: {r.text[:200]}")
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


def to_sharadar(sym: str) -> str:
    """yfinance share-class style BRK-B -> Sharadar BRK.B."""
    return sym.replace("-", ".")


def universe() -> list[str]:
    syms = set()
    if MEMBERSHIP.exists():
        syms |= set(pd.read_parquet(MEMBERSHIP)["symbol"].astype(str))
    if PANEL.exists():
        syms |= set(pd.read_parquet(PANEL, columns=["symbol"])["symbol"].astype(str))
    return sorted(s for s in syms if s and not s.startswith("^"))


def resolve_aliases(missing: list[str], key: str) -> tuple[dict, pd.DataFrame]:
    """Map S&P-era symbols to Sharadar's current (post-delisting) tickers
    via the delisted TICKERS table's relatedtickers field."""
    if not missing:
        return {}, pd.DataFrame()
    print(f"resolving {len(missing)} symbols through the delisted TICKERS table...")
    delisted = _get("tickers", key, table="stocks", isdelisted="Y",
                    fields="ticker,name,relatedtickers,firstpricedate,lastpricedate,category")
    print(f"  delisted stocks listed by Sharadar: {len(delisted):,}")
    want = {to_sharadar(s): s for s in missing}
    alias: dict = {}
    if len(delisted):
        for _, row in delisted.sort_values("lastpricedate").iterrows():
            rel = str(row.get("relatedtickers") or "")
            for tok in rel.split():
                if tok in want and want[tok] not in alias:
                    alias[want[tok]] = row["ticker"]      # later rows override earlier (latest wins)
                elif tok in want:
                    alias[want[tok]] = row["ticker"]
    print(f"  resolved {len(alias)} aliases, e.g. "
          f"{dict(list(alias.items())[:6])}")
    return alias, delisted


def main(sample: bool, refresh: bool) -> int:
    key = get_key("SHARADAR_API_KEY") or ("test-api-key" if sample else "")
    if not key:
        print("SHARADAR_API_KEY not set (env or config_keys.py). Use --sample for the free DJIA-30 smoke test.")
        return 1
    key_mode = "sample" if key == "test-api-key" else "subscription"
    syms = ["AAPL", "MSFT"] if sample else universe()
    print(f"universe: {len(syms)} symbols | key: {key_mode}")

    have = set()
    if OUT_PRICES.exists() and not refresh:
        prior = json.load(open(MANIFEST)) if MANIFEST.exists() else {}
        if prior.get("key_mode") == "sample" and key_mode != "sample":
            print("stored data came from the sample key -> refetching everything")
            refresh = True
        else:
            stored = pd.read_parquet(OUT_PRICES, columns=["symbol"])
            have = set(stored["symbol"].unique())
    todo = [s for s in syms if s not in have]
    print(f"stored: {len(have)}, to fetch: {len(todo)}")

    # 1) ticker metadata: direct hits
    meta_frames = []
    for i in range(0, len(syms), BATCH):
        chunk = [to_sharadar(s) for s in syms[i:i + BATCH]]
        try:
            meta_frames.append(_get("tickers", key, ticker=",".join(chunk)))
        except Exception as exc:  # noqa: BLE001
            print(f"  [tickers] batch {i // BATCH}: {exc}")
        time.sleep(SLEEP)
    meta = pd.concat([m for m in meta_frames if len(m)], ignore_index=True) if meta_frames else pd.DataFrame()
    if len(meta) and "table" in meta.columns:
        meta = meta[meta["table"] == "stocks"]
    direct = set(meta["ticker"]) if len(meta) else set()
    unresolved = [s for s in syms if to_sharadar(s) not in direct]
    print(f"tickers table: {len(meta)} direct hits, {len(unresolved)} symbols not under their S&P-era code")

    # 2) alias resolution through the delisted list
    alias, delisted = resolve_aliases(unresolved, key)
    if len(delisted):
        extra = delisted[delisted["ticker"].isin(set(alias.values()))]
        meta = pd.concat([meta, extra], ignore_index=True)
    if len(meta):
        meta = meta.drop_duplicates("ticker")
        meta.to_parquet(OUT_TICKERS, index=False)
        n_del = int((meta["isdelisted"].astype(str).str.upper() == "Y").sum()) if "isdelisted" in meta else -1
        print(f"tickers stored: {len(meta)} rows | delisted flagged: {n_del}")
    still_missing = [s for s in unresolved if s not in alias]

    # 3) prices, one call per symbol (paged if > 10k rows)
    frames, failed = [], []
    for n, sym in enumerate(todo, 1):
        tk = alias.get(sym, to_sharadar(sym))
        if sym in still_missing:
            continue
        try:
            df = _get("stocks", key, ticker=tk, **{"from": START})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{n}/{len(todo)}] {sym}: {exc}")
            failed.append(sym)
            continue
        if df.empty:
            failed.append(sym)
        else:
            df["symbol"] = sym
            frames.append(df)
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {sym} ({tk}): {len(df)} rows")
            if frames:  # incremental save
                new = pd.concat(frames, ignore_index=True)
                if OUT_PRICES.exists() and not refresh:
                    old = pd.read_parquet(OUT_PRICES)
                    new = pd.concat([old[~old["symbol"].isin(new["symbol"])], new], ignore_index=True)
                new.to_parquet(OUT_PRICES, index=False)
                frames, refresh = [], False
        time.sleep(SLEEP)

    total = pd.read_parquet(OUT_PRICES) if OUT_PRICES.exists() else pd.DataFrame()
    manifest = {
        "fetched_at": datetime.now().isoformat(), "start": START, "key_mode": key_mode,
        "universe": len(syms),
        "stored_symbols": int(total["symbol"].nunique()) if len(total) else 0,
        "rows": int(len(total)),
        "aliases": alias, "unresolved": still_missing, "fetch_failed": failed,
        "date_min": str(total["date"].min()) if len(total) else None,
        "date_max": str(total["date"].max()) if len(total) else None,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nstored symbols: {manifest['stored_symbols']}/{len(syms)}, rows: {manifest['rows']:,}, "
          f"{manifest['date_min']} .. {manifest['date_max']}")
    print(f"aliases resolved: {len(alias)} | unresolved: {len(still_missing)} {still_missing[:15]}"
          f"{'...' if len(still_missing) > 15 else ''} | fetch failed: {len(failed)}")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="free DJIA-30 smoke test with test-api-key")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.sample, a.refresh))
