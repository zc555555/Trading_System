"""Repair EDGAR companyfacts caches that hold only a stub of a company's
history because the ticker now points at a different CIK.

Two causes seen on 2026-09-04:
  * re-domiciled / re-organised issuers get a NEW CIK (Exxon Mobil: 34088
    -> 2115436 in 2026) and SEC's company_tickers.json maps the ticker to
    the new one, whose companyfacts starts this year;
  * a departed member's ticker was re-used by an unrelated small company
    (APC, KG, MON, POM, Q, ...).

For every cache whose facts span less than MIN_YEARS, the CIK that Sharadar's
TICKERS table records for the symbol (secfilings link) is fetched from
data.sec.gov and its facts are MERGED into the cache (union per tag/unit,
deduplicated on accession + period + value). When Sharadar names the same
CIK nothing can be done. The cache keeps `cik` (current) and gains `ciks`.

Usage:  python data/fix_edgar_cache_ciks.py [--dry]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent
CACHE = DATA / "edgar_cache"
TICKERS = DATA / "sharadar_tickers.parquet"
PRICES = DATA / "sharadar_prices.parquet"
UA = {"User-Agent": "Trading_System research zhaochenwu666@gmail.com"}
MIN_YEARS = 3.0


def span_years(facts: dict) -> float:
    dates = []
    for tag in ("Assets", "NetIncomeLoss", "StockholdersEquity"):
        for r in facts.get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", []):
            dates.append(r["filed"])
    if not dates:
        return 0.0
    d = pd.to_datetime(pd.Series(sorted(set(dates))))
    return float((d.iloc[-1] - d.iloc[0]).days / 365.25)


def sharadar_ciks() -> dict[str, list[str]]:
    pairs = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
    tick = pd.read_parquet(TICKERS)
    cik = tick["secfilings"].astype(str).str.extract(r"CIK=(\d+)", expand=False)
    t2c = {t: c for t, c in zip(tick["ticker"], cik) if isinstance(c, str)}
    out: dict[str, list[str]] = {}
    for s, t in pairs.itertuples(index=False):
        for key in (s, t):
            c = t2c.get(key)
            if c and c not in out.setdefault(s, []):
                out[s].append(c)
    return out


def merge_facts(base: dict, extra: dict) -> int:
    added = 0
    for taxo, tags in extra.get("facts", {}).items():
        bt = base.setdefault("facts", {}).setdefault(taxo, {})
        for tag, node in tags.items():
            bn = bt.setdefault(tag, {"label": node.get("label"), "description": node.get("description"), "units": {}})
            for unit, rows in node.get("units", {}).items():
                have = bn["units"].setdefault(unit, [])
                seen = {(r.get("accn"), r.get("start"), r.get("end"), r.get("val")) for r in have}
                for r in rows:
                    k = (r.get("accn"), r.get("start"), r.get("end"), r.get("val"))
                    if k not in seen:
                        have.append(r)
                        seen.add(k)
                        added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    alt = sharadar_ciks()
    fixed, unfixable = [], []
    for f in sorted(CACHE.glob("*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if span_years(d.get("facts", {})) >= MIN_YEARS:
            continue
        sym = f.stem
        cur = str(d.get("cik", "")).lstrip("0")
        cands = [c.lstrip("0") for c in alt.get(sym, []) if c.lstrip("0") != cur]
        if not cands:
            unfixable.append((sym, cur))
            continue
        total = 0
        for c in cands:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json"
            r = requests.get(url, headers=UA, timeout=120)
            time.sleep(0.15)
            if r.status_code != 200:
                print(f"  {sym}: CIK {c} -> HTTP {r.status_code}")
                continue
            total += merge_facts(d, r.json())
        d["ciks"] = sorted(set([cur] + cands))
        if total and not args.dry:
            json.dump(d, open(f, "w", encoding="utf-8"))
        fixed.append((sym, cur, cands, total, round(span_years(d.get("facts", {})), 1)))
        print(f"  {sym}: current CIK {cur}, merged {cands}: +{total} facts, span now {fixed[-1][-1]} y")
    print(f"repaired {len(fixed)} caches; unfixable (Sharadar knows no other CIK): {unfixable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
