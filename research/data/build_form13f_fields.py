"""Point-in-time institutional ownership for the mining DSL from the SEC
Form 13F data sets (quarterly zips in research/data/form13f/, downloaded
from https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets).

What is measured, per (symbol, quarter end P):
    inst_shares   shares held by all 13F filers (common stock rows only:
                  SSHPRNAMTTYPE == SH, no PUTCALL), original 13F-HR
                  filings, first filing per (manager, period)
    inst_holders  number of distinct managers holding the stock
    top5_share    share of inst_shares held by the five largest holders

Point-in-time rule: a quarter's holdings are used only once the statutory
deadline has passed (45 days after quarter end, DEADLINE_DAYS), and only
filings made on or before that deadline count; late filers and amendments
are ignored rather than allowed to update history. The daily fields
(mining/aux_fields.attach_form13f) therefore become usable from the first
session after P + 45 days and are carried at most one quarter.

Issuer mapping: INFOTABLE.CUSIP -> symbol through Sharadar's TICKERS
``cusips`` column (space-separated, historical CUSIPs included).

Output data/form13f_quarterly.parquet, one row per (symbol, period_end):
    period_end, usable_from (= P + DEADLINE_DAYS), inst_shares, inst_holders,
    top5_share, n_filings (13F-HR originals counted for the period)

Usage:  python data/build_form13f_fields.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
ZIPS = DATA / "form13f"
OUT = DATA / "form13f_quarterly.parquet"
TICKERS = DATA / "sharadar_tickers.parquet"
PRICES = DATA / "sharadar_prices.parquet"
DEADLINE_DAYS = 45
CHUNK = 500_000


def cusip_map() -> dict[str, str]:
    pairs = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
    t2s = dict(zip(pairs["ticker"], pairs["symbol"]))
    tick = pd.read_parquet(TICKERS, columns=["ticker", "cusips"])
    m: dict[str, str] = {}
    for t, cus in zip(tick["ticker"], tick["cusips"]):
        if t not in t2s or not isinstance(cus, str):
            continue
        for c in cus.split():
            c = c.strip().upper()
            if len(c) >= 8:
                m.setdefault(c[:9] if len(c) >= 9 else c, t2s[t])
    return m


def _read(z: zipfile.ZipFile, name: str, cols: list[str], chunksize: int | None = None):
    return pd.read_csv(io.BytesIO(z.read(name)), sep="\t", usecols=cols, dtype=str, encoding="latin-1",
                       quoting=3, on_bad_lines="skip", chunksize=chunksize)


def process_zip(path: Path, cusips: dict[str, str]) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"].copy()
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub = sub.dropna(subset=["filed", "period"])
    sub["deadline"] = sub["period"] + pd.Timedelta(days=DEADLINE_DAYS)
    sub = sub[sub["filed"] <= sub["deadline"]]                       # only filings the market had by the deadline
    sub = sub.sort_values("filed").drop_duplicates(["CIK", "period"], keep="first")
    acc = sub.set_index("ACCESSION_NUMBER")[["CIK", "period"]]
    parts = []
    for chunk in _read(z, "INFOTABLE.tsv", ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"], CHUNK):
        chunk = chunk[chunk["ACCESSION_NUMBER"].isin(acc.index)]
        chunk = chunk[(chunk["SSHPRNAMTTYPE"] == "SH") & chunk["PUTCALL"].isna()]
        chunk["symbol"] = chunk["CUSIP"].str.upper().str[:9].map(cusips)
        chunk = chunk[chunk["symbol"].notna()]
        if chunk.empty:
            continue
        chunk["shares"] = pd.to_numeric(chunk["SSHPRNAMT"], errors="coerce").fillna(0.0)
        parts.append(chunk[["ACCESSION_NUMBER", "symbol", "shares"]])
    if not parts:
        return pd.DataFrame()
    h = pd.concat(parts, ignore_index=True).merge(acc, left_on="ACCESSION_NUMBER", right_index=True)
    per_mgr = h.groupby(["symbol", "period", "CIK"], as_index=False)["shares"].sum()
    per_mgr = per_mgr[per_mgr["shares"] > 0]
    g = per_mgr.groupby(["symbol", "period"])
    out = g.agg(inst_shares=("shares", "sum"), inst_holders=("CIK", "nunique")).reset_index()
    top5 = (per_mgr.sort_values("shares", ascending=False).groupby(["symbol", "period"]).head(5)
            .groupby(["symbol", "period"])["shares"].sum().rename("top5"))
    out = out.merge(top5, on=["symbol", "period"], how="left")
    out["top5_share"] = out["top5"] / out["inst_shares"]
    n_filings = sub.groupby("period").size().rename("n_filings")
    out = out.merge(n_filings, left_on="period", right_index=True, how="left")
    return out.drop(columns=["top5"]).rename(columns={"period": "period_end"})


def main():
    cusips = cusip_map()
    print(f"CUSIP map: {len(cusips)} CUSIPs -> {len(set(cusips.values()))} symbols")
    frames = []
    for path in sorted(ZIPS.glob("*_form13f.zip")):
        try:
            df = process_zip(path, cusips)
        except zipfile.BadZipFile:
            print(f"  {path.name}: not a zip (incomplete download), skipped")
            continue
        print(f"  {path.name}: {len(df):,} symbol-quarters", flush=True)
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # a period can appear in two consecutive datasets (filings straddle the dataset window): combine them
    out = (out.groupby(["symbol", "period_end"], as_index=False)
              .agg(inst_shares=("inst_shares", "sum"), inst_holders=("inst_holders", "sum"),
                   top5_share=("top5_share", "max"), n_filings=("n_filings", "sum")))
    out["usable_from"] = out["period_end"] + pd.Timedelta(days=DEADLINE_DAYS)
    out = out.sort_values(["symbol", "period_end"]).reset_index(drop=True)
    for c in ("inst_shares", "inst_holders", "top5_share", "n_filings"):
        out[c] = out[c].astype("float64")
    out.to_parquet(OUT, index=False)
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} symbol-quarters "
          f"({out.period_end.min().date()} .. {out.period_end.max().date()}) -> {OUT.name}")
    print("median holders per symbol-quarter:", out.inst_holders.median())


if __name__ == "__main__":
    sys.exit(main())
