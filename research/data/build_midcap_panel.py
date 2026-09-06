"""Mid-cap research panel and its auxiliary tables (research only).

  1. stocks_with_time_windows_midcap.parquet  OHLCV for every mid-cap member
     from sharadar_prices_midcap.parquet, in the S&P panel's convention:
     open/high/low/close scaled by closeadj/close (dividend + split adjusted),
     tz-aware dates, one row per (date, symbol). No production features yet:
     the screen stage needs only OHLCV + auxiliary fields; the full stage
     (walk-forward with the incumbent factor groups) needs the production
     feature chain, which is a later step.
  2. edgar_fields_midcap.parquet  share counts from the SEC frames table
     (data/build_midcap_universe.py): one row per (symbol, quarter end) with
     filed = quarter end + FRAMES_LAG_DAYS -- the frames API has no filing
     date, so the statutory 10-Q deadline stands in for it (documented
     approximation; the S&P table uses true filing dates).
  3. regsho_short_volume_midcap.parquet  FINRA Reg SHO daily short volume
     filtered to the mid-cap members (the raw daily files cover every symbol).

Usage:  python data/build_midcap_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
PRICES = DATA / "sharadar_prices_midcap.parquet"
MEMBERSHIP = DATA / "midcap_membership.parquet"
TICKERS_ALL = DATA / "sharadar_tickers_all.parquet"
FRAMES = DATA / "frames_shares.parquet"
RAW_REGSHO = DATA / "regsho"
OUT_PANEL = DATA / "stocks_with_time_windows_midcap.parquet"
OUT_EDGAR = DATA / "edgar_fields_midcap.parquet"
OUT_REGSHO = DATA / "regsho_short_volume_midcap.parquet"
FRAMES_LAG_DAYS = 45


def build_panel() -> pd.DataFrame:
    px = pd.read_parquet(PRICES)
    members = set(pd.read_parquet(MEMBERSHIP)["symbol"].astype(str))
    px = px[px["symbol"].isin(members)].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize("America/New_York")
    scale = (px["closeadj"] / px["close"]).where(px["close"] > 0, 1.0)
    for c in ("open", "high", "low", "close"):
        px[c] = px[c].astype(float) * scale
    px["volume"] = px["volume"].astype(float)
    out = (px[["date", "symbol", "open", "high", "low", "close", "volume"]]
           .dropna(subset=["close"]).sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"]).reset_index(drop=True))
    out.to_parquet(OUT_PANEL, index=False)
    print(f"panel: {out.symbol.nunique():,} symbols, {len(out):,} rows ({out.date.min().date()} .. {out.date.max().date()}) -> {OUT_PANEL.name}")
    return out


def build_edgar_fields(members: set) -> pd.DataFrame:
    t = pd.read_parquet(TICKERS_ALL, columns=["ticker", "secfilings"])
    t = t[t["ticker"].isin(members)].copy()
    t["cik"] = pd.to_numeric(t["secfilings"].astype(str).str.extract(r"CIK=(\d+)", expand=False), errors="coerce")
    cik2sym = {int(c): s for s, c in zip(t["ticker"], t["cik"]) if pd.notna(c)}
    fr = pd.read_parquet(FRAMES)
    fr = fr[fr["cik"].isin(cik2sym)].copy()
    fr["symbol"] = fr["cik"].astype(int).map(cik2sym)
    out = pd.DataFrame({"symbol": fr["symbol"], "filed": pd.to_datetime(fr["end"]) + pd.Timedelta(days=FRAMES_LAG_DAYS),
                        "form": "frames+45d", "shares_out": fr["val"].astype(float), "src": "dei-frames"})
    out = out.sort_values(["symbol", "filed"]).reset_index(drop=True)
    out.to_parquet(OUT_EDGAR, index=False)
    print(f"edgar fields (frames): {out.symbol.nunique():,} symbols, {len(out):,} quarter rows -> {OUT_EDGAR.name}")
    return out


def build_regsho(members: set) -> pd.DataFrame:
    frames = []
    for f in sorted(RAW_REGSHO.glob("CNMSshvol*.txt")):
        if f.stat().st_size == 0:
            continue
        df = pd.read_csv(f, sep="|", dtype={"Symbol": str}, usecols=["Date", "Symbol", "ShortVolume", "TotalVolume"])
        frames.append(df[df["Symbol"].isin(members)])
    out = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date", "Symbol": "symbol", "ShortVolume": "short_volume", "TotalVolume": "total_volume"})
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d")
    for c in ("short_volume", "total_volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.groupby(["date", "symbol"], as_index=False).agg(short_volume=("short_volume", "sum"), total_volume=("total_volume", "sum"))
    out["short_ratio_day"] = (out["short_volume"] / out["total_volume"]).where(out["total_volume"] > 0)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(OUT_REGSHO, index=False)
    print(f"regsho: {out.symbol.nunique():,} symbols, {len(out):,} rows -> {OUT_REGSHO.name}")
    return out


def main():
    panel = build_panel()
    members = set(panel["symbol"].unique())
    build_edgar_fields(members)
    if RAW_REGSHO.exists():
        build_regsho(members)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
