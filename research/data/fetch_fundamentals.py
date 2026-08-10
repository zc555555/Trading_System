"""SEC EDGAR XBRL fundamentals backfill (merged-B item 5, wave 2).

Downloads companyfacts for every S&P-member symbol and extracts a small
robust concept set into a long point-in-time table. Every value carries
its FIRST filing date -- the day the market actually learned it -- which
is the availability timestamp downstream features must respect.

Concept extraction rules:
- tag priority lists (XBRL tag heterogeneity across filers)
- one row per unique fiscal period, FIRST `filed` wins (later filings
  repeat prior periods as comparatives; amendments come later by
  definition)
- flow concepts (revenue, income, cashflow): quarterly durations
  (60-120d) kept directly; Q4 derived as FY - (Q1+Q2+Q3) matched on the
  `fy` field, dated by the 10-K's filing date (standard technique --
  many filers only report annual flows in the 10-K)
- instant concepts (assets, equity, shares): taken as-is

Output: data/fundamentals.parquet
    [symbol, concept, start, end, fy, fp, filed, val]
Raw JSONs cached under data/edgar_cache/ (gitignored) so re-parsing
does not re-download. SEC fair use: UA with contact, <=10 req/s.

Usage:
    python data/fetch_fundamentals.py --probe AAPL
    python data/fetch_fundamentals.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent
CACHE = DATA_DIR / "edgar_cache"
OUT = DATA_DIR / "fundamentals.parquet"
MEMBERSHIP = DATA_DIR / "sp500_membership.parquet"
UA = {"User-Agent": "stock_predict research zhaochenwu666@gmail.com"}
SLEEP_S = 0.15

# concept -> (namespace, [tag priority list], kind)
CONCEPTS = {
    "revenue": ("us-gaap", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax"], "flow"),
    "net_income": ("us-gaap", ["NetIncomeLoss", "ProfitLoss"], "flow"),
    "cogs": ("us-gaap", [
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"], "flow"),
    "gross_profit": ("us-gaap", ["GrossProfit"], "flow"),
    "op_cashflow": ("us-gaap", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "flow"),
    "assets": ("us-gaap", ["Assets"], "instant"),
    "equity": ("us-gaap", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "instant"),
    "shares_out": ("dei", ["EntityCommonStockSharesOutstanding"], "instant"),
}


def norm_symbol(t: str) -> str:
    return t.strip().upper().replace("-", "").replace(".", "")


def cik_map() -> dict[str, int]:
    m = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=UA, timeout=30).json()
    return {norm_symbol(v["ticker"]): v["cik_str"] for v in m.values()}


def fetch_facts(symbol: str, cik: int) -> dict | None:
    cache_file = CACHE / f"{symbol}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            CACHE.mkdir(exist_ok=True)
            cache_file.write_text(r.text, encoding="utf-8")
            return r.json()
        except Exception as e:
            print(f"  [retry {attempt+1}] {symbol}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def _pick_rows(facts: dict, namespace: str, tags: list[str],
               unit_pref=("USD", "shares")) -> list[dict]:
    """UNION across the tag priority list. Companies switch tags over time
    (e.g. SalesRevenueNet -> RevenueFromContractWithCustomer... at ASC 606
    adoption in 2018); first-tag-wins silently drops the older history.
    Earlier tags in the list win per-period dedup downstream via `filed`.
    """
    ns = facts.get("facts", {}).get(namespace, {})
    rows: list[dict] = []
    for tag in tags:
        node = ns.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        for u in unit_pref:
            if u in units and units[u]:
                rows.extend(units[u])
                break
    return rows


def extract_symbol(symbol: str, facts: dict) -> pd.DataFrame:
    frames = []
    for concept, (namespace, tags, kind) in CONCEPTS.items():
        rows = _pick_rows(facts, namespace, tags)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if df.empty or "end" not in df or "filed" not in df:
            continue
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df = df.dropna(subset=["end", "filed", "val"])
        if kind == "flow" and "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
            df = df.dropna(subset=["start"])
            df["dur"] = (df["end"] - df["start"]).dt.days
            q = df[(df["dur"] >= 60) & (df["dur"] <= 120)].copy()
            fy = df[(df["dur"] >= 330) & (df["dur"] <= 380)].copy()
            # first filing per unique period
            q = (q.sort_values("filed")
                   .drop_duplicates(["start", "end"], keep="first"))
            fy = (fy.sort_values("filed")
                    .drop_duplicates(["start", "end"], keep="first"))

            # Cashflow statements (and some filers generally) report
            # year-to-date cumulatives in 10-Qs (Q2 = 6mo, Q3 = 9mo).
            # Difference consecutive same-start cumulatives into quarters;
            # each derived quarter is dated by the LATER filing.
            ytd = df[(df["dur"] >= 150) & (df["dur"] <= 290)].copy()
            if len(ytd):
                pool = (pd.concat([q, ytd], ignore_index=True)
                          .sort_values(["start", "end", "filed"])
                          .drop_duplicates(["start", "end"], keep="first"))
                diffs = []
                for s, grp in pool.groupby("start"):
                    grp = grp.sort_values("end")
                    prev = None
                    for _, r in grp.iterrows():
                        if prev is not None:
                            gap = (r["end"] - prev["end"]).days
                            if 60 <= gap <= 120:
                                diffs.append({
                                    "start": prev["end"], "end": r["end"],
                                    "fy": r.get("fy"), "fp": "Qd",
                                    "filed": max(r["filed"], prev["filed"]),
                                    "val": r["val"] - prev["val"]})
                        prev = r
                if diffs:
                    q = (pd.concat([q, pd.DataFrame(diffs)], ignore_index=True)
                           .sort_values("filed")
                           .drop_duplicates(["end"], keep="first"))
            # derive Q4 = FY - (Q1+Q2+Q3) matched inside the FY window
            derived = []
            for _, a in fy.iterrows():
                in_fy = q[(q["start"] >= a["start"]) & (q["end"] <= a["end"])]
                if len(in_fy) == 3:
                    derived.append({
                        "start": in_fy["end"].max(), "end": a["end"],
                        "fy": a.get("fy"), "fp": "Q4d",
                        "filed": a["filed"],
                        "val": a["val"] - in_fy["val"].sum()})
            out = pd.concat([q, pd.DataFrame(derived)], ignore_index=True) \
                if derived else q
        else:
            out = (df.sort_values("filed")
                     .drop_duplicates(["end"], keep="first")).copy()
            out["start"] = pd.NaT
        out["concept"] = concept
        out["symbol"] = symbol
        keep = ["symbol", "concept", "start", "end", "fy", "fp", "filed", "val"]
        for c in keep:
            if c not in out.columns:
                out[c] = None
        frames.append(out[keep])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(probe: str | None):
    members = sorted(pd.read_parquet(MEMBERSHIP)["symbol"].unique())
    print(f"symbols: {len(members)} S&P members (ever)")
    cmap = cik_map()

    if probe:
        cik = cmap.get(norm_symbol(probe))
        facts = fetch_facts(probe, cik)
        df = extract_symbol(probe, facts)
        print(df.groupby("concept").agg(rows=("val", "size"),
                                        first=("end", "min"),
                                        last=("end", "max")).to_string())
        return

    frames, missing = [], []
    t0 = time.time()
    for i, sym in enumerate(members, 1):
        cik = cmap.get(norm_symbol(sym))
        if cik is None:
            missing.append(sym)
            continue
        facts = fetch_facts(sym, cik)
        if facts is None:
            missing.append(sym)
            continue
        df = extract_symbol(sym, facts)
        if len(df):
            frames.append(df)
        if i % 40 == 0 or i == len(members):
            print(f"  [{i}/{len(members)}] {sym} "
                  f"({time.time() - t0:.0f}s, {len(missing)} missing)")
        time.sleep(SLEEP_S)

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"\nsaved: {len(out):,} rows, {out['symbol'].nunique()} symbols "
          f"-> {OUT.name}")
    print(f"missing (no CIK/facts): {len(missing)}: {missing[:12]}")
    cov = out.groupby("concept")["symbol"].nunique()
    print("\nper-concept symbol coverage:")
    print(cov.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None)
    args = ap.parse_args()
    main(args.probe)
