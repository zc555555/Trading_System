"""Point-in-time share counts and filing calendar from the local EDGAR cache.

    python data/build_edgar_fields.py

Reads every data/edgar_cache/<SYMBOL>.json (SEC companyfacts, fetched by
fetch_fundamentals.py) and writes data/edgar_fields.parquet with one row per
(symbol, filed) filing event:

    symbol, filed (YYYY-MM-DD), form (10-Q / 10-K / ...), shares_out

`shares_out` is dei:EntityCommonStockSharesOutstanding as reported on the
cover page of that filing (an instant value; the latest `end` within the
filing wins). `filed` is the day the market could first know it, so an
as-of join on `filed` is point-in-time by construction. The mining DSL's
auxiliary fields (mining/aux_fields.py) derive from this file:

    marketcap   = close * shares_out / 1e6      (USD millions)
    turnover    = volume / shares_out
    filing_days = sessions since the last 10-Q/10-K filing usable today

No network access: rerun fetch_fundamentals.py first to refresh the cache.
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CACHE = DATA_DIR / "edgar_cache"
OUT = DATA_DIR / "edgar_fields.parquet"
MANIFEST = DATA_DIR / "edgar_fields_manifest.json"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-QT", "20-F", "40-F"}


FALLBACK_TAG = "WeightedAverageNumberOfSharesOutstandingBasic"   # us-gaap, per-period duration value
MAX_QUARTER_DAYS = 100


def _recs(tag_obj) -> list[dict]:
    out = []
    for recs in (tag_obj or {}).get("units", {}).values():
        out.extend(recs)
    return out


def rows_for(path: Path) -> list[dict]:
    """One row per filing. Prefer the cover-page instant count (dei); for a
    filing without it, fall back to the basic weighted-average share count of
    the shortest period reported in that filing (us-gaap)."""
    j = json.loads(path.read_text(encoding="utf-8"))
    sym = path.stem
    facts = j.get("facts", {})
    out = []
    seen = set()
    for r in _recs(facts.get("dei", {}).get("EntityCommonStockSharesOutstanding")):
        filed, val, form = r.get("filed"), r.get("val"), str(r.get("form", ""))
        if not filed or val is None or form not in FORMS:
            continue
        out.append({"symbol": sym, "filed": filed, "form": form, "end": r.get("end"),
                    "shares_out": float(val), "src": "dei"})
        seen.add(filed)
    fb = {}
    for r in _recs(facts.get("us-gaap", {}).get(FALLBACK_TAG)):
        filed, val, form = r.get("filed"), r.get("val"), str(r.get("form", ""))
        if not filed or val is None or form not in FORMS or filed in seen:
            continue
        try:
            days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
        except Exception:
            continue
        if days > MAX_QUARTER_DAYS and filed in fb and fb[filed]["days"] <= MAX_QUARTER_DAYS:
            continue                                   # a quarterly figure already found for this filing
        cur = fb.get(filed)
        if cur is None or (days <= MAX_QUARTER_DAYS < cur["days"]) or (r["end"] > cur["end"] and (days <= MAX_QUARTER_DAYS) == (cur["days"] <= MAX_QUARTER_DAYS)):
            fb[filed] = {"symbol": sym, "filed": filed, "form": form, "end": r["end"],
                         "shares_out": float(val), "src": "waso", "days": days}
    out.extend({k: v for k, v in x.items() if k != "days"} for x in fb.values())
    return out


def build() -> pd.DataFrame:
    files = sorted(glob.glob(str(CACHE / "*.json")))
    rows = []
    for p in files:
        rows.extend(rows_for(Path(p)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # several share-class rows can share one filing: sum classes reported at
    # the same (symbol, filed, end, src); then keep the latest `end` per filing
    df = df.groupby(["symbol", "filed", "form", "end", "src"], as_index=False)["shares_out"].sum()
    df = df.sort_values(["symbol", "filed", "end"]).drop_duplicates(["symbol", "filed"], keep="last")
    df = df[df["shares_out"] > 0]
    return df[["symbol", "filed", "form", "shares_out", "src"]].reset_index(drop=True)


def main() -> int:
    df = build()
    if df.empty:
        print("no shares_out facts found in", CACHE)
        return 1
    df.to_parquet(OUT, index=False)
    man = {"built_at": datetime.now().isoformat(), "rows": int(len(df)), "symbols": int(df["symbol"].nunique()),
           "filed_min": str(df["filed"].min()), "filed_max": str(df["filed"].max()),
           "forms": df["form"].value_counts().to_dict(), "sources": df["src"].value_counts().to_dict()}
    MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"saved {OUT.name}: {len(df):,} filings, {df['symbol'].nunique()} symbols, "
          f"{man['filed_min']}..{man['filed_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
