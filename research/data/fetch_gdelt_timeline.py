"""GDELT daily tone/volume backfill via DOC 2.0 timeline modes.

Unlike artlist (250 articles/query, short history), the timeline modes
return a DAILY TIME SERIES for the whole query span in one request:
    mode=timelinetone    -> average tone of matching coverage per day
    mode=timelinevolraw  -> raw count of matching articles per day
Coverage: Jan 2017 onward. Two requests per symbol => ~600 requests for
the full 302-symbol universe (~15 min at 1.2s spacing).

Company-name mapping: yfinance longName/shortName, cached to
company_names.json, merged with the curated COMPANY_NAMES variants in
fetch_gdelt_news.py. Query = OR of quoted name variants.

Output: gdelt_daily.parquet  [date, symbol, gdelt_tone, gdelt_articles]
Incremental: symbols already present in the output are skipped, so the
job can be re-run after interruptions.

Usage:
    python data/fetch_gdelt_timeline.py --test AAPL     # single-symbol probe
    python data/fetch_gdelt_timeline.py                 # full backfill
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config  # noqa: E402
from data.fetch_gdelt_news import COMPANY_NAMES  # noqa: E402

API = "https://api.gdeltproject.org/api/v2/doc/doc"
START = "20170101000000"          # timeline modes cover 2017+
DATA_DIR = Path(__file__).parent
NAMES_CACHE = DATA_DIR / "company_names.json"
OUTPUT = DATA_DIR / "gdelt_daily.parquet"
# GDELT advertises one request per 5s, but during high-traffic periods the
# effective throttle is harsher and 429 bursts cascade. Generous spacing +
# long backoff + a cooldown after total failure keeps throughput steady.
SLEEP_S = 8.0
FAIL_COOLDOWN_S = 60
GENERIC = {"inc", "corp", "corporation", "company", "co", "ltd", "plc",
           "group", "holdings", "class", "the"}


def load_company_names(symbols: list[str]) -> dict[str, list[str]]:
    """Resolve each symbol to quoted-name variants, cached on disk."""
    cache: dict[str, list[str]] = {}
    if NAMES_CACHE.exists():
        cache = json.loads(NAMES_CACHE.read_text(encoding="utf-8"))

    missing = [s for s in symbols if s not in cache]
    if missing:
        import yfinance as yf
        print(f"[names] resolving {len(missing)} symbols via yfinance...")
        for i, sym in enumerate(missing, 1):
            variants = list(COMPANY_NAMES.get(sym, []))
            try:
                info = yf.Ticker(sym).info
                for key in ("longName", "shortName"):
                    name = info.get(key)
                    if name and name not in variants:
                        variants.append(name)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")
            if not variants:
                variants = [sym]
            cache[sym] = variants
            if i % 25 == 0 or i == len(missing):
                NAMES_CACHE.write_text(json.dumps(cache, indent=1),
                                       encoding="utf-8")
                print(f"  [names] {i}/{len(missing)} cached")
            time.sleep(0.3)
    return {s: cache[s] for s in symbols}


def build_query(variants: list[str]) -> str:
    """OR of quoted name variants; strip stray quotes; drop 1-word generic."""
    phrases = []
    for v in variants[:4]:
        v = re.sub(r'["()]', "", v).strip()
        if not v:
            continue
        words = [w for w in v.split() if w.lower().rstrip(".,") not in GENERIC]
        cleaned = " ".join(words) if words else v
        if len(cleaned) < 3:
            continue
        phrases.append(f'"{cleaned}"')
    if not phrases:
        phrases = [f'"{variants[0]}"']
    return phrases[0] if len(phrases) == 1 else "(" + " OR ".join(dict.fromkeys(phrases)) + ")"


def fetch_timeline(query: str, mode: str, retries: int = 4) -> pd.Series | None:
    params = {"query": query, "mode": mode, "format": "json",
              "startdatetime": START,
              "enddatetime": pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
            data = r.json()
            series = data.get("timeline", [])
            if not series:
                return None
            points = series[0].get("data", [])
            out = {pd.to_datetime(p["date"], format="%Y%m%dT%H%M%SZ"): p["value"]
                   for p in points}
            return pd.Series(out).sort_index()
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"    [retry {attempt + 1}] {mode}: {e} (wait {wait}s)")
            time.sleep(wait)
    return None


def fetch_symbol(sym: str, variants: list[str]) -> pd.DataFrame | None:
    query = build_query(variants)
    tone = fetch_timeline(query, "timelinetone")
    time.sleep(SLEEP_S)
    vol = fetch_timeline(query, "timelinevolraw")
    time.sleep(SLEEP_S)
    if tone is None and vol is None:
        return None
    df = pd.DataFrame({"gdelt_tone": tone, "gdelt_articles": vol})
    df.index.name = "date"
    df = df.reset_index()
    df["symbol"] = sym
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default=None,
                    help="Probe a single symbol and print stats")
    args = ap.parse_args()

    config = load_config()
    symbols = config["data"]["symbols"]
    names = load_company_names([args.test] if args.test else symbols)

    if args.test:
        sym = args.test
        print(f"[test] {sym}: query = {build_query(names[sym])}")
        df = fetch_symbol(sym, names[sym])
        if df is None:
            print("[test] NO DATA")
            return
        print(df.describe())
        print(f"dates: {df['date'].min()} .. {df['date'].max()}  rows={len(df)}")
        print(df.tail(3).to_string(index=False))
        return

    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        frames.append(existing)
        done = set(existing["symbol"].unique())
        print(f"[resume] {len(done)} symbols already fetched")

    todo = [s for s in symbols if s not in done]
    print(f"[fetch] {len(todo)} symbols to go (~{len(todo) * 2 * (SLEEP_S + 1):.0f}s)")
    t0 = time.time()
    for i, sym in enumerate(todo, 1):
        df = fetch_symbol(sym, names[sym])
        status = f"{len(df)} days" if df is not None else "NO DATA"
        print(f"  [{i}/{len(todo)}] {sym}: {status}")
        if df is not None:
            frames.append(df)
        else:
            # Total failure usually means we are in a throttle spiral;
            # cool down so subsequent symbols recover.
            print(f"    [cooldown] {FAIL_COOLDOWN_S}s after total failure")
            time.sleep(FAIL_COOLDOWN_S)
        if i % 20 == 0 or i == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(OUTPUT, index=False)
    if frames:
        final = pd.concat(frames, ignore_index=True)
        final.to_parquet(OUTPUT, index=False)
        print(f"\n[done] {final['symbol'].nunique()} symbols, "
              f"{len(final):,} rows -> {OUTPUT.name} "
              f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
