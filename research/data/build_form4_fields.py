"""Daily insider-trading events for the mining DSL from the SEC's
Insider Transactions data sets (Forms 3/4/5, quarterly zips in
research/data/form345/, downloaded from
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets).

Only original Form 4 filings (DOCUMENT_TYPE == "4") and open-market
transactions in the non-derivative table count:
    TRANS_CODE "P" (open-market or private purchase)  -> buy
    TRANS_CODE "S" (open-market or private sale)      -> sell
Option exercises, grants, gifts, tax withholding etc. are excluded, as in
Lakonishok-Lee (2001) and Cohen-Malloy-Pomorski (2012).

The issuer is mapped by CIK (ISSUERCIK) to our panel symbol through the
EDGAR companyfacts cache (research/data/edgar_cache/<symbol>.json carries
the CIK) plus Sharadar's secfilings CIK for delisted members; the trading
symbol printed on the form is only a fallback.

Output research/data/form4_daily.parquet, one row per (symbol, filed):
    filed        SEC filing date (the day the market could first know it;
                 usable from the next session, see mining/aux_fields)
    buy_shares   shares bought (P) in the filings of that day
    sell_shares  shares sold (S)
    buy_value    sum(shares * price), USD
    sell_value   idem for sales
    n_buyers     distinct reporting owners with a P transaction
    n_sellers    distinct reporting owners with an S transaction

Usage:  python data/build_form4_fields.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
ZIPS = DATA / "form345"
OUT = DATA / "form4_daily.parquet"
CACHE = DATA / "edgar_cache"
TICKERS = DATA / "sharadar_tickers.parquet"
PRICES = DATA / "sharadar_prices.parquet"


def cik_map() -> dict[str, str]:
    """CIK (10-digit string) -> panel symbol."""
    m: dict[str, str] = {}
    if TICKERS.exists() and PRICES.exists():
        pairs = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
        t2s = dict(zip(pairs["ticker"], pairs["symbol"]))
        tick = pd.read_parquet(TICKERS)
        if "secfilings" in tick.columns:
            cik = tick["secfilings"].astype(str).str.extract(r"CIK=(\d+)", expand=False)
            for t, c in zip(tick["ticker"], cik):
                if isinstance(c, str) and t in t2s:
                    m[c.zfill(10)] = t2s[t]
    for f in sorted(CACHE.glob("*.json")):          # companyfacts cache wins (it is our own symbol)
        try:
            c = json.load(open(f, encoding="utf-8")).get("cik")
        except Exception:
            continue
        if c is not None:
            m[str(c).zfill(10)] = f.stem
    return m


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), sep="\t", usecols=cols, dtype=str,
                       encoding="latin-1", quoting=3, on_bad_lines="skip")


def process_zip(path: Path, ciks: dict[str, str]) -> pd.DataFrame:
    z = zipfile.ZipFile(path)
    sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"] == "4"].copy()
    sub["symbol"] = sub["ISSUERCIK"].astype(str).str.zfill(10).map(ciks)
    sub = sub[sub["symbol"].notna()]
    if sub.empty:
        return pd.DataFrame()
    tr = _read(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"]) & tr["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])].copy()
    if tr.empty:
        return pd.DataFrame()
    own = _read(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK"])
    tr["shares"] = pd.to_numeric(tr["TRANS_SHARES"], errors="coerce").fillna(0.0)
    tr["price"] = pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce").fillna(0.0)
    tr["value"] = tr["shares"] * tr["price"]
    tr["side"] = tr["TRANS_CODE"].map({"P": "buy", "S": "sell"})
    tr = tr.merge(sub[["ACCESSION_NUMBER", "FILING_DATE", "symbol"]], on="ACCESSION_NUMBER", how="inner")
    tr = tr.merge(own.drop_duplicates(), on="ACCESSION_NUMBER", how="left")
    tr["filed"] = pd.to_datetime(tr["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    tr = tr[tr["filed"].notna()]
    g = tr.groupby(["symbol", "filed", "side"])
    agg = g.agg(shares=("shares", "sum"), value=("value", "sum"), n=("RPTOWNERCIK", "nunique")).reset_index()
    wide = agg.pivot_table(index=["symbol", "filed"], columns="side", values=["shares", "value", "n"], fill_value=0.0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ("shares_buy", "shares_sell", "value_buy", "value_sell", "n_buy", "n_sell"):
        if c not in wide.columns:
            wide[c] = 0.0
    return wide.rename(columns={"shares_buy": "buy_shares", "shares_sell": "sell_shares",
                                "value_buy": "buy_value", "value_sell": "sell_value",
                                "n_buy": "n_buyers", "n_sell": "n_sellers"})


def main():
    ciks = cik_map()
    print(f"CIK map: {len(ciks)} CIKs -> {len(set(ciks.values()))} symbols")
    frames = []
    for path in sorted(ZIPS.glob("*_form345.zip")):
        df = process_zip(path, ciks)
        print(f"  {path.name}: {len(df):,} symbol-days", flush=True)
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = (out.groupby(["symbol", "filed"], as_index=False)
              .sum(numeric_only=True)
              .sort_values(["symbol", "filed"]).reset_index(drop=True))
    for c in ("buy_shares", "sell_shares", "buy_value", "sell_value", "n_buyers", "n_sellers"):
        out[c] = out[c].astype("float64")
    out.to_parquet(OUT, index=False)
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} symbol-days "
          f"({out.filed.min().date()} .. {out.filed.max().date()}) -> {OUT.name}")
    print(f"buy days: {(out.buy_shares > 0).sum():,}  sell days: {(out.sell_shares > 0).sum():,}")


if __name__ == "__main__":
    sys.exit(main())
