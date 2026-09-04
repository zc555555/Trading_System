"""FINRA Reg SHO daily short sale volume (consolidated NMS file per session).

Source: https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt
(no registration; one pipe-delimited file per trading day, published the
same evening). The CDN hosts files from 2017-12-29 on; earlier dates answer
403 and the old regsho.finra.org host now redirects to a catalog page, so
the history starts where FINRA's short-interest API does. Columns: Date|Symbol|ShortVolume|ShortExemptVolume|
TotalVolume|Market. TotalVolume is the volume reported to FINRA's trade
reporting facilities (off-exchange), not consolidated tape volume, so the
ratio ShortVolume / TotalVolume is the meaningful field, not the levels.

Idempotent: raw files are cached in data/regsho/ and only missing sessions
are fetched; the output parquet is rebuilt from the cache every run.

Output data/regsho_short_volume.parquet: date, symbol, short_volume,
total_volume, short_ratio_day (= short / total, NaN when total is 0).

Usage:  python data/fetch_regsho_short_volume.py [--start 2017-12-29] [--sleep 0.15]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent
RAW = DATA / "regsho"
OUT = DATA / "regsho_short_volume.parquet"
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
MEMBERS = DATA / "sharadar_prices.parquet"


def sessions(start: date, end: date) -> list[date]:
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def fetch(start: date, end: date, sleep: float) -> tuple[int, int]:
    RAW.mkdir(exist_ok=True)
    sess = requests.Session()
    got = missing = 0
    for d in sessions(start, end):
        ymd = d.strftime("%Y%m%d")
        target = RAW / f"CNMSshvol{ymd}.txt"
        if target.exists():
            continue
        r = sess.get(URL.format(ymd=ymd), timeout=60)
        if r.status_code == 200 and r.text.startswith("Date|Symbol"):
            target.write_text(r.text, encoding="utf-8")
            got += 1
        else:                                   # holiday, pre-2018 (403: not hosted) or not yet published
            missing += 1
            if d < end - timedelta(days=3):     # settled: remember as empty so re-runs skip it
                target.write_text("", encoding="utf-8")
        time.sleep(sleep)
    return got, missing


def build() -> pd.DataFrame:
    members = set(pd.read_parquet(MEMBERS, columns=["symbol"])["symbol"].unique())
    frames = []
    for f in sorted(RAW.glob("CNMSshvol*.txt")):
        if f.stat().st_size == 0:
            continue
        df = pd.read_csv(f, sep="|", dtype={"Symbol": str}, usecols=["Date", "Symbol", "ShortVolume", "TotalVolume"])
        df = df[df["Symbol"].isin(members)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"Date": "date", "Symbol": "symbol", "ShortVolume": "short_volume", "TotalVolume": "total_volume"})
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d")
    for c in ("short_volume", "total_volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["short_ratio_day"] = (out["short_volume"] / out["total_volume"]).where(out["total_volume"] > 0)
    out = (out.groupby(["date", "symbol"], as_index=False)
              .agg(short_volume=("short_volume", "sum"), total_volume=("total_volume", "sum")))
    out["short_ratio_day"] = (out["short_volume"] / out["total_volume"]).where(out["total_volume"] > 0)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-12-29", help="first session; FINRA's CDN hosts nothing earlier (403)")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()
    t0 = time.time()
    got, missing = fetch(date.fromisoformat(args.start), date.today(), args.sleep)
    print(f"fetched {got} new files, {missing} absent (holidays/unpublished), {time.time() - t0:.0f}s", flush=True)
    out = build()
    out.to_parquet(OUT, index=False)
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} rows ({out.date.min().date()} .. {out.date.max().date()}) -> {OUT.name}")
    print(f"median daily short ratio: {out.short_ratio_day.median():.3f}")


if __name__ == "__main__":
    sys.exit(main())
