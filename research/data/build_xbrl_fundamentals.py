"""Point-in-time fundamentals for the mining DSL from the EDGAR companyfacts
cache (research/data/edgar_cache/<symbol>.json, SEC XBRL "frames" API
material: every reported fact with the filing it came from).

Point-in-time discipline
  * a fact is known from its own `filed` date; when the same period is
    reported again later (comparatives, restatements) the EARLIEST filing
    wins, so a restated number never leaks back before its restatement;
  * flows (revenue, net income, ...) are trailing-twelve-month sums of
    quarterly values known at the filing: quarterly facts are 3-month
    durations, and the fourth quarter (never reported as 3 months) is the
    fiscal-year fact minus the three quarters inside it;
  * stocks (assets, equity, ...) are the latest instant known at the filing;
  * assets one year earlier is the instant closest to period end - 1 year.

Output data/xbrl_fundamentals.parquet, one row per (symbol, filed) with
the state known after that filing (previous values carried forward):
    period_end   latest fiscal period end reported by this filing
    revenue_ttm, gp_ttm, ni_ttm, ocf_ttm, capex_ttm, rd_ttm, opinc_ttm   USD
    assets, equity, liabilities, ltdebt, cash, assets_1y                 USD
The ratio fields the DSL sees (book_to_market, gross_profitability,
asset_growth, accruals, ...) are formed daily in mining/aux_fields.

Usage:  python data/build_xbrl_fundamentals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent
CACHE = DATA / "edgar_cache"
OUT = DATA / "xbrl_fundamentals.parquet"

FLOW_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"],
    "gp": ["GrossProfit"],
    "ni": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "rd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "opinc": ["OperatingIncomeLoss"],
}
STOCK_TAGS = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "ltdebt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F", "10-KT", "10-QT")
QUARTER = (75, 105)          # duration in days of a 3-month fact
YEAR = (350, 380)            # of a 12-month fact
STALE_DAYS = 400             # a metric whose latest period is older than this vs the filing's period is unknown


def _facts(gaap: dict, tags: list[str]) -> pd.DataFrame:
    """All USD facts of the first tags that exist, as a frame with
    start/end/val/filed; a later tag only fills periods the earlier tags
    did not report."""
    frames = []
    for tag in tags:
        node = gaap.get(tag)
        if not node or "USD" not in node.get("units", {}):
            continue
        df = pd.DataFrame(node["units"]["USD"])
        if df.empty:
            continue
        df = df[df["form"].isin(FORMS)] if "form" in df.columns else df
        if "start" not in df.columns:
            df["start"] = None
        frames.append(df[["start", "end", "val", "filed"]].assign(tag=tag))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "tag"])
    out = pd.concat(frames, ignore_index=True)
    out["end"] = pd.to_datetime(out["end"])
    out["start"] = pd.to_datetime(out["start"])
    out["filed"] = pd.to_datetime(out["filed"])
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out = out.dropna(subset=["val", "end", "filed"])
    # earliest filing per period (PIT); first tag in preference order wins ties
    out["_rank"] = out["tag"].map({t: i for i, t in enumerate(tags)})
    out = out.sort_values(["end", "start", "filed", "_rank"]).drop_duplicates(["start", "end"], keep="first")
    return out.drop(columns=["_rank"])


def quarterly_series(fl: pd.DataFrame) -> pd.DataFrame:
    """3-month values per fiscal quarter end with the filing date they became
    known; Q4 derived from the fiscal-year fact minus its three quarters."""
    fl = fl[fl["start"].notna()].copy()
    fl["dur"] = (fl["end"] - fl["start"]).dt.days
    q = fl[fl["dur"].between(*QUARTER)][["start", "end", "val", "filed"]].copy()
    y = fl[fl["dur"].between(*YEAR)][["start", "end", "val", "filed"]].copy()
    derived = []
    for _, yr in y.iterrows():
        inside = q[(q["start"] >= yr["start"] - pd.Timedelta(days=10)) & (q["end"] <= yr["end"] - pd.Timedelta(days=60))]
        if len(inside) == 3:
            derived.append({"start": inside["end"].max(), "end": yr["end"],
                            "val": yr["val"] - inside["val"].sum(),
                            "filed": max(yr["filed"], inside["filed"].max())})
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    q = q.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return q[["end", "val", "filed"]].reset_index(drop=True)


def ttm_events(q: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month sums as (filed, period_end, value) events: at
    each quarter's filing, the four most recent consecutive quarters known."""
    if len(q) < 4:
        return pd.DataFrame(columns=["filed", "period_end", "val"])
    q = q.sort_values("end").reset_index(drop=True)
    rows = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3:i + 1]
        span = (w["end"].iloc[-1] - w["end"].iloc[0]).days
        if 240 <= span <= 300:                                   # four consecutive quarters
            rows.append({"filed": w["filed"].max(), "period_end": w["end"].iloc[-1], "val": float(w["val"].sum())})
    return pd.DataFrame(rows, columns=["filed", "period_end", "val"])


def stock_events(fl: pd.DataFrame) -> pd.DataFrame:
    s = fl[fl["start"].isna()][["end", "val", "filed"]].copy()
    return s.sort_values(["end", "filed"]).drop_duplicates("end", keep="first").rename(columns={"end": "period_end"})


def build_symbol(symbol: str, gaap: dict) -> pd.DataFrame:
    events = []                                                 # (filed, period_end, metric, value)
    flows = {}
    for metric, tags in FLOW_TAGS.items():
        fl = _facts(gaap, tags)
        if fl.empty:
            continue
        ev = ttm_events(quarterly_series(fl))
        flows[metric] = ev
        for _, r in ev.iterrows():
            events.append((r["filed"], r["period_end"], f"{metric}_ttm", r["val"]))
    if "gp" not in flows and "revenue" in flows and "cogs" in flows:   # gross profit = revenue - cogs
        m = flows["revenue"].merge(flows["cogs"], on="period_end", suffixes=("_r", "_c"))
        for _, r in m.iterrows():
            events.append((max(r["filed_r"], r["filed_c"]), r["period_end"], "gp_ttm", r["val_r"] - r["val_c"]))
    stocks = {}
    for metric, tags in STOCK_TAGS.items():
        fl = _facts(gaap, tags)
        if fl.empty:
            continue
        ev = stock_events(fl)
        stocks[metric] = ev
        for _, r in ev.iterrows():
            events.append((r["filed"], r["period_end"], metric, r["val"]))
    if "assets" in stocks:                                      # assets one year before each period end
        a = stocks["assets"].sort_values("period_end")
        for _, r in a.iterrows():
            target = r["period_end"] - pd.Timedelta(days=365)
            prev = a[(a["period_end"] - target).abs() <= pd.Timedelta(days=45)]
            prev = prev[prev["filed"] <= r["filed"]]
            if len(prev):
                events.append((r["filed"], r["period_end"], "assets_1y", float(prev.iloc[-1]["val"])))
    if not events:
        return pd.DataFrame()
    ev = pd.DataFrame(events, columns=["filed", "period_end", "metric", "val"])
    # the state known after each filing date: latest period per metric, carried
    # forward only while that period is recent (a tag a company stopped using
    # must not be carried for years, e.g. a bank's LongTermDebt last filed 2014)
    ev = ev.sort_values(["metric", "filed", "period_end"]).drop_duplicates(["metric", "filed"], keep="last")
    wide = ev.pivot(index="filed", columns="metric", values="val").sort_index().ffill()
    ends = ev.pivot(index="filed", columns="metric", values="period_end").sort_index().ffill()
    pe = ev.groupby("filed")["period_end"].max().reindex(wide.index).ffill()
    for m in wide.columns:
        stale = (pe - ends[m]).dt.days > STALE_DAYS
        wide.loc[stale, m] = np.nan
    wide.insert(0, "period_end", pe.to_numpy())
    wide.insert(0, "symbol", symbol)
    return wide.reset_index()


def main():
    frames = []
    files = sorted(CACHE.glob("*.json"))
    for i, f in enumerate(files, 1):
        try:
            gaap = json.load(open(f, encoding="utf-8")).get("facts", {}).get("us-gaap", {})
        except Exception as e:
            print(f"  {f.stem}: unreadable ({e})")
            continue
        df = build_symbol(f.stem, gaap)
        if not df.empty:
            frames.append(df)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    cols = ["symbol", "filed", "period_end", "revenue_ttm", "gp_ttm", "ni_ttm", "ocf_ttm", "capex_ttm", "rd_ttm",
            "opinc_ttm", "assets", "equity", "liabilities", "ltdebt", "cash", "assets_1y"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values(["symbol", "filed"]).reset_index(drop=True)
    for c in cols[3:]:
        out[c] = out[c].astype("float64")
    out.to_parquet(OUT, index=False)
    cov = {c: round(float(out[c].notna().mean()), 3) for c in cols[3:]}
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} filing rows ({out.filed.min().date()} .. {out.filed.max().date()}) -> {OUT.name}")
    print("row coverage:", cov)


if __name__ == "__main__":
    sys.exit(main())
