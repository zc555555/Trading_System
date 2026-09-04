"""Daily English-Wikipedia page views of each member company's article
(retail attention proxy; Wikimedia REST API, no key, User-Agent required).

Mapping: data/wikidata_tickers.csv (Wikidata items with a NYSE/NASDAQ
ticker, P414/P249, and an enwiki sitelink; built by a SPARQL query) joined
to the member universe on the ticker, with Sharadar's ticker as an alias.
Ticker re-use can map an old member to a newer company's article; the
matched table is written to data/wikipedia_articles.csv for inspection.

Endpoint (per article, daily, all agents = user only):
  https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/<title>/daily/<start>/<end>
Data exist from 2015-07-01. Numbers for a UTC day are final after the day
ends and are never restated, so causality is a next-session rule
(mining/aux_fields.attach_wikipedia).

Output data/wikipedia_pageviews.parquet: date (UTC calendar day), symbol,
views. Idempotent: an article already in the parquet is only extended
from its last stored day.

Usage:  python data/fetch_wikipedia_pageviews.py [--sleep 0.2] [--start 20150701]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

DATA = Path(__file__).resolve().parent
WIKIDATA = DATA / "wikidata_tickers.csv"
ARTICLES = DATA / "wikipedia_articles.csv"
OUT = DATA / "wikipedia_pageviews.parquet"
PRICES = DATA / "sharadar_prices.parquet"
UA = {"User-Agent": "Trading_System research zhaochenwu666@gmail.com"}
URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/daily/{start}/{end}"


def _norm(s: str) -> str:
    return s.upper().replace(".", "").replace("-", "")


def article_map() -> pd.DataFrame:
    wd = pd.read_csv(WIKIDATA, dtype=str)
    pairs = pd.read_parquet(PRICES, columns=["symbol", "ticker"]).drop_duplicates()
    alias: dict[str, str] = {}
    for s, t in pairs.itertuples(index=False):
        alias[_norm(s)] = s
        if isinstance(t, str):
            alias.setdefault(_norm(t), s)
    wd["symbol"] = wd["ticker"].map(lambda t: alias.get(_norm(str(t))))
    m = wd[wd["symbol"].notna()].drop_duplicates("symbol")[["symbol", "ticker", "exchange", "article"]]
    m.to_csv(ARTICLES, index=False)
    return m


def fetch_article(title: str, start: str, end: str, sess: requests.Session) -> pd.DataFrame:
    url = URL.format(title=quote(title, safe=""), start=start, end=end)
    r = None
    for attempt in range(4):
        r = sess.get(url, headers=UA, timeout=60)
        if r.status_code == 200:
            items = r.json().get("items", [])
            return pd.DataFrame({"date": [pd.Timestamp(i["timestamp"][:8]) for i in items],
                                 "views": [float(i["views"]) for i in items]})
        if r.status_code == 404:                    # no data in range (new article) -> empty
            return pd.DataFrame(columns=["date", "views"])
        time.sleep(1.5 * (attempt + 1))
    print(f"  {title}: HTTP {getattr(r, 'status_code', None)}", flush=True)
    return pd.DataFrame(columns=["date", "views"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--start", default="20150701")
    args = ap.parse_args()
    m = article_map()
    print(f"articles: {len(m)} member symbols mapped", flush=True)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["date", "symbol", "views"])
    last = old.groupby("symbol")["date"].max() if len(old) else pd.Series(dtype="datetime64[ns]")
    end = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    sess = requests.Session()
    frames, t0 = [old], time.time()
    for i, (sym, title) in enumerate(zip(m["symbol"], m["article"]), 1):
        start = args.start
        if sym in last.index:
            start = (last[sym] + pd.Timedelta(days=1)).strftime("%Y%m%d")
            if start > end:
                continue
        df = fetch_article(title, start, end, sess)
        if len(df):
            frames.append(df.assign(symbol=sym))
        if i % 100 == 0:
            print(f"  {i}/{len(m)}  {time.time() - t0:.0f}s", flush=True)
        time.sleep(args.sleep)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    out["views"] = out["views"].astype("float64")
    out.to_parquet(OUT, index=False)
    print(f"saved: {out.symbol.nunique()} symbols, {len(out):,} rows ({out.date.min().date()} .. {out.date.max().date()}) -> {OUT.name}")


if __name__ == "__main__":
    sys.exit(main())
