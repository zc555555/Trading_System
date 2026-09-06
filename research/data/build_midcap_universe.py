"""Point-in-time US mid-cap research universe (Track P / mining only; the
live book stays S&P 500).

Definition (RULEBOOK "Universes"): at each month end, rank every US
domestic common stock (Sharadar SEP, NYSE/NASDAQ/NYSEMKT, primary class)
by market capitalisation = month-end close x the latest SEC-reported common
shares outstanding whose quarter end is at least SHARES_LAG_DAYS old (a
proxy for the filing lag; the SEC frames API carries no filing date). A
stock ENTERS the universe when its rank lies in [ENTER_LO, ENTER_HI] and
LEAVES when it drops below EXIT_LO (grew into the large caps) or above
EXIT_HI (shrank away), or is delisted. Hysteresis keeps monthly churn low.

Survivorship: delisted names are ranked with everyone else while they
trade, so the universe contains the dead. Names without SEC share data
(mostly foreign filers and tiny companies) cannot be ranked and are absent.

Inputs (cached, refreshed with --refresh):
  data/sharadar_tickers_all.parquet   full Sharadar stock ticker table
  data/sep_monthend.parquet           SEP rows for every month end (one API call per month)
  data/frames_shares.parquet          dei:EntityCommonStockSharesOutstanding per CIK and quarter
Output:
  data/midcap_membership.parquet      symbol (Sharadar ticker), start, end   (same schema as sp500_membership)
  data/midcap_universe_stats.json

Usage:  python data/build_midcap_universe.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from data.vendor_keys import get_key  # noqa: E402
import fetch_sharadar_prices as fs  # noqa: E402

DATA = Path(__file__).resolve().parent
TICKERS_ALL = DATA / "sharadar_tickers_all.parquet"
SEP_ME = DATA / "sep_monthend.parquet"
FRAMES = DATA / "frames_shares.parquet"
OUT = DATA / "midcap_membership.parquet"
STATS = DATA / "midcap_universe_stats.json"
UA = {"User-Agent": "Trading_System research zhaochenwu666@gmail.com"}

FIRST_MONTH = "2013-12"
ENTER_LO, ENTER_HI = 501, 1400
EXIT_LO, EXIT_HI = 450, 1650
SHARES_LAG_DAYS = 45
MIN_PRICE = 1.0


def candidates() -> pd.DataFrame:
    t = pd.read_parquet(TICKERS_ALL)
    cat = t["category"].astype(str)
    keep = (cat.isin(["Domestic Common Stock", "Domestic Common Stock Primary Class"])
            & t["exchange"].isin(["NYSE", "NASDAQ", "NYSEMKT"]))
    t = t[keep].copy()
    t["cik"] = t["secfilings"].astype(str).str.extract(r"CIK=(\d+)", expand=False)
    t["cik"] = pd.to_numeric(t["cik"], errors="coerce")
    t["lastpricedate"] = pd.to_datetime(t["lastpricedate"], errors="coerce")
    t["firstpricedate"] = pd.to_datetime(t["firstpricedate"], errors="coerce")
    t = t[(t["lastpricedate"] >= "2014-06-01") & (t["firstpricedate"] <= "2026-06-01")]
    return t[["ticker", "cik", "name", "exchange", "isdelisted", "lastpricedate", "firstpricedate", "sector", "scalemarketcap"]]


def month_ends() -> list[pd.Timestamp]:
    return list(pd.date_range(FIRST_MONTH, pd.Timestamp.today().normalize(), freq="BME"))


def fetch_sep_monthend(key: str, refresh: bool) -> pd.DataFrame:
    have = pd.read_parquet(SEP_ME) if SEP_ME.exists() and not refresh else pd.DataFrame(columns=["ticker", "date", "close", "closeunadj", "volume"])
    done = set(pd.to_datetime(have["date"]).dt.normalize()) if len(have) else set()
    frames = [have]
    for me in month_ends():
        if me in done:
            continue
        d = me
        rows = pd.DataFrame()
        for back in range(0, 5):                       # holiday at month end: step back a day at a time
            try:
                rows = fs._get("stocks", key, date=(me - pd.Timedelta(days=back)).strftime("%Y-%m-%d"))
            except Exception as e:                     # noqa: BLE001
                print(f"  {me.date()}: {e}"); rows = pd.DataFrame()
            if len(rows):
                d = me - pd.Timedelta(days=back)
                break
            time.sleep(0.3)
        if rows.empty:
            print(f"  {me.date()}: no rows", flush=True)
            continue
        rows = rows[["ticker", "date", "close", "closeunadj", "volume"]].copy()
        rows["date"] = me                              # keyed by the calendar month end
        rows["session"] = d
        frames.append(rows)
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(SEP_ME, index=False)
    return out


def fetch_frames(refresh: bool) -> pd.DataFrame:
    have = pd.read_parquet(FRAMES) if FRAMES.exists() and not refresh else pd.DataFrame(columns=["cik", "end", "val", "frame"])
    done = set(have["frame"]) if len(have) else set()
    frames = [have]
    start = pd.Period("2013Q3", freq="Q")
    end = pd.Timestamp.today().to_period("Q")
    p = start
    while p <= end:
        fr = f"CY{p.year}Q{p.quarter}I"
        if fr not in done:
            url = f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{fr}.json"
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200:
                data = r.json().get("data", [])
                df = pd.DataFrame(data)[["cik", "end", "val"]] if data else pd.DataFrame(columns=["cik", "end", "val"])
                df["frame"] = fr
                frames.append(df)
                print(f"  {fr}: {len(df):,} companies", flush=True)
            else:
                print(f"  {fr}: HTTP {r.status_code}", flush=True)
            time.sleep(0.15)
        p += 1
    out = pd.concat(frames, ignore_index=True)
    out["end"] = pd.to_datetime(out["end"])
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce")
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out = out.dropna(subset=["cik", "end", "val"]).sort_values(["cik", "end"]).drop_duplicates(["cik", "end"], keep="last")
    out.to_parquet(FRAMES, index=False)
    return out


def rank_table(cand: pd.DataFrame, sep: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    """Month-end market-cap rank per candidate ticker."""
    sep = sep[sep["ticker"].isin(set(cand["ticker"]))].merge(cand[["ticker", "cik"]], on="ticker", how="left")
    sep = sep.dropna(subset=["cik"]).copy()
    sep["cik"] = sep["cik"].astype("int64")
    sh = sh.copy(); sh["cik"] = sh["cik"].astype("int64")
    sep["cutoff"] = sep["date"] - pd.Timedelta(days=SHARES_LAG_DAYS)
    sh = sh.sort_values("end")
    sep = sep.sort_values("cutoff")
    m = pd.merge_asof(sep, sh[["cik", "end", "val"]].rename(columns={"val": "shares", "end": "shares_end"}),
                      left_on="cutoff", right_on="shares_end", by="cik", direction="backward")
    m = m[m["shares_end"] >= m["date"] - pd.Timedelta(days=400)]        # no share count older than ~13 months
    m["px"] = m["closeunadj"].where(m["closeunadj"] > 0, m["close"])
    m = m[(m["px"] >= MIN_PRICE) & (m["shares"] > 0)].copy()
    m["mcap"] = m["px"] * m["shares"]
    m["rank"] = m.groupby("date")["mcap"].rank(ascending=False, method="first")
    return m[["ticker", "date", "mcap", "rank"]].sort_values(["ticker", "date"])


def memberships(r: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    last_px = dict(zip(cand["ticker"], cand["lastpricedate"]))
    rows = []
    for tk, g in r.groupby("ticker", sort=False):
        g = g.sort_values("date")
        inside, start = False, None
        for d, rk in zip(g["date"], g["rank"]):
            if not inside and ENTER_LO <= rk <= ENTER_HI:
                inside, start = True, d + pd.Timedelta(days=1)
            elif inside and (rk < EXIT_LO or rk > EXIT_HI):
                rows.append({"symbol": tk, "start": start, "end": d}); inside = False
        if inside:
            lp = last_px.get(tk)
            end = pd.NaT if (pd.isna(lp) or lp >= pd.Timestamp.today() - pd.Timedelta(days=45)) else lp
            rows.append({"symbol": tk, "start": start, "end": end})
    out = pd.DataFrame(rows)
    return out.sort_values(["symbol", "start"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    key = get_key("SHARADAR_API_KEY")
    if not key:
        raise SystemExit("SHARADAR_API_KEY missing")
    cand = candidates()
    print(f"candidates: {len(cand):,} tickers ({int(cand['cik'].notna().sum()):,} with a CIK)")
    sep = fetch_sep_monthend(key, args.refresh)
    print(f"month-end prices: {sep['date'].nunique()} month ends, {len(sep):,} rows")
    sh = fetch_frames(args.refresh)
    print(f"shares frames: {sh['frame'].nunique()} quarters, {sh['cik'].nunique():,} CIKs")
    r = rank_table(cand, sep, sh)
    print(f"ranked: {r['ticker'].nunique():,} tickers, {r['date'].nunique()} month ends; "
          f"median ranked per month {r.groupby('date').size().median():.0f}")
    mem = memberships(r, cand)
    mem.to_parquet(OUT, index=False)
    per_year = {}
    for y in range(2014, pd.Timestamp.today().year + 1):
        d = pd.Timestamp(f"{y}-06-30")
        per_year[y] = int(((mem["start"] <= d) & (mem["end"].isna() | (mem["end"] >= d))).sum())
    stats = {"tickers": int(mem["symbol"].nunique()), "intervals": int(len(mem)),
             "delisted_members": int(mem["symbol"].isin(cand.loc[cand["isdelisted"].astype(str).str.upper() == "Y", "ticker"]).sum()),
             "members_on_june_30": per_year, "band": [ENTER_LO, ENTER_HI, EXIT_LO, EXIT_HI], "shares_lag_days": SHARES_LAG_DAYS}
    STATS.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    print(f"saved -> {OUT.name}")


if __name__ == "__main__":
    sys.exit(main())
