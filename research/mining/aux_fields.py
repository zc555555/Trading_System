"""Auxiliary DSL fields beyond OHLCV, attached to a (date, symbol) panel.

Source: research/data/edgar_fields.parquet (data/build_edgar_fields.py), one
row per SEC filing with the cover-page share count -- point-in-time by
construction, because `filed` is the day the market could first know it.
A filing on session a becomes usable at session a + 1 (filings arrive
during the day; the same "second session at-or-after" convention the PEAD
experiment used for announcements).

    shares_out   latest usable cover-page share count (as-of join on filed)
    marketcap    close * shares_out / 1e6, USD millions
    turnover     volume / shares_out: fraction of shares traded in the session
    filing_days  sessions since the last usable 10-Q/10-K filing; 0 on the
                 first usable session; NaN before a symbol's first filing

Only past filings of the same symbol and same-day price/volume are read, so
the fields are causal like the rest of the DSL; the tests add a future
filing and check that past values do not move. A stale share count is
carried at most SHARES_MAX_AGE sessions (a company that stops filing drops
out rather than being priced on old numbers).

If the source file is missing the columns are absent and any expression
using them is refused by the DSL; nothing is silently zero-filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
EDGAR = DATA / "edgar_fields.parquet"
GDELT = DATA / "gdelt_daily.parquet"
FORM4 = DATA / "form4_daily.parquet"
SHORT = DATA / "finra_short_interest.parquet"
AUX_FIELDS = ("marketcap", "turnover", "filing_days", "news_tone", "news_articles",
              "insider_buys", "insider_sells", "insider_net_frac", "short_ratio", "days_to_cover")
SHARES_MAX_AGE = 130          # ~two quarters of sessions
SHORT_INTEREST_LAG = 10       # sessions after settlement before FINRA's figure is public (~7 business days)
SHORT_MAX_AGE = 40            # a short-interest figure is carried at most this many sessions


def _naive_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s


def attach_filings(df: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """As-of join of the filing calendar onto the panel by session ordinal:
    the last filing whose usable session (filed session + 1) <= this row's
    session gives shares_out and filing_days."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ord_of = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_ord"] = ord_of.reindex(naive.to_numpy()).to_numpy()

    fl = filings[["symbol", "filed", "shares_out"]].copy()
    fl["filed"] = pd.to_datetime(fl["filed"]).dt.normalize()
    a = np.searchsorted(sessions, fl["filed"].to_numpy(dtype="datetime64[ns]"), side="left")
    keep = a < len(sessions)
    fl = fl[keep].copy()
    fl["eff_ord"] = a[keep] + 1                      # usable from the session after the filing
    fl = fl.sort_values(["eff_ord"]).drop_duplicates(["symbol", "eff_ord"], keep="last")

    left = out[["symbol", "_ord"]].reset_index().sort_values("_ord")
    merged = pd.merge_asof(left, fl[["symbol", "eff_ord", "shares_out"]].sort_values("eff_ord"),
                           left_on="_ord", right_on="eff_ord", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    age = (merged["_ord"] - merged["eff_ord"]).astype(float)
    # beyond SHARES_MAX_AGE the company has stopped filing (or the cache
    # ends): both the share count and the calendar become unknown, not stale
    fresh = age <= SHARES_MAX_AGE
    out["shares_out"] = merged["shares_out"].astype(float).where(fresh).reindex(out.index).to_numpy()
    out["filing_days"] = age.where(fresh).reindex(out.index).to_numpy()
    out["marketcap"] = out["close"].astype(float) * out["shares_out"] / 1e6
    out["turnover"] = out["volume"].astype(float) / out["shares_out"]
    for c in ("marketcap", "turnover"):
        out.loc[~np.isfinite(out[c]), c] = np.nan
    return out.drop(columns=["_ord", "shares_out"])


def attach_news(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """GDELT daily tone / article count (calendar days) onto the panel.

    news_tone      article-weighted mean tone of the calendar days that became
                   usable at this session; NaN when no article
    news_articles  article count over those days; 0 when the symbol has news
                   coverage but no article, NaN when the symbol is absent from
                   the source altogether

    Causality: the aggregate for calendar day D is complete only after D ends,
    so D's news is usable from the FIRST SESSION STRICTLY AFTER D (a Friday's,
    Saturday's and Sunday's news all land on Monday). This is one session more
    conservative than the news experiment's same-day convention."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    n = news[["symbol", "date", "gdelt_tone", "gdelt_articles"]].copy()
    n["date"] = pd.to_datetime(n["date"]).dt.normalize()
    n["gdelt_tone"] = pd.to_numeric(n["gdelt_tone"], errors="coerce")
    n["gdelt_articles"] = pd.to_numeric(n["gdelt_articles"], errors="coerce").fillna(0.0)
    pos = np.searchsorted(sessions, n["date"].to_numpy(dtype="datetime64[ns]"), side="right")
    keep = pos < len(sessions)
    n = n[keep].copy()
    n["eff"] = sessions[pos[keep]]
    n["_w"] = n["gdelt_tone"] * n["gdelt_articles"]
    agg = n.groupby(["symbol", "eff"]).agg(_w=("_w", "sum"), news_articles=("gdelt_articles", "sum")).reset_index()
    agg["news_tone"] = (agg["_w"] / agg["news_articles"]).where(agg["news_articles"] > 0)
    key = pd.DataFrame({"symbol": out["symbol"].to_numpy(), "eff": naive.to_numpy()}, index=out.index)
    merged = key.merge(agg[["symbol", "eff", "news_tone", "news_articles"]], on=["symbol", "eff"], how="left")
    covered = out["symbol"].isin(set(n["symbol"])).to_numpy()
    tone = merged["news_tone"].to_numpy(dtype=float)
    arts = merged["news_articles"].to_numpy(dtype=float)
    arts = np.where(np.isnan(arts) & covered, 0.0, arts)
    out["news_tone"] = np.where(covered, tone, np.nan)
    out["news_articles"] = np.where(covered, arts, np.nan)
    return out


def _shares(out: pd.DataFrame) -> np.ndarray:
    """Shares outstanding implied by attach_filings' marketcap (NaN when the
    EDGAR fields are absent or stale)."""
    if "marketcap" not in out.columns:
        return np.full(len(out), np.nan)
    sh = out["marketcap"].to_numpy(dtype=float) * 1e6 / out["close"].to_numpy(dtype=float)
    return np.where(np.isfinite(sh) & (sh > 0), sh, np.nan)


def attach_form4(df: pd.DataFrame, events: pd.DataFrame, known_through=None) -> pd.DataFrame:
    """Open-market insider trades (Form 4, data/build_form4_fields.py).

    insider_buys      distinct insiders whose purchase filings became usable
                      at this session; 0 when the symbol is covered and quiet,
                      NaN when the symbol has no CIK mapping at all
    insider_sells     idem for sales
    insider_net_frac  (shares bought - shares sold) / shares outstanding;
                      NaN when the share count is unknown

    A filing on calendar day D is usable from the first session strictly
    after D (Form 4s arrive during and after the session). The source is a
    quarterly bulk dataset, so sessions after its last covered day
    (known_through, default = the last filing date in `events`) are
    unknown, not quiet: every field is NaN there."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ev = events[["symbol", "filed", "buy_shares", "sell_shares", "n_buyers", "n_sellers"]].copy()
    ev["filed"] = pd.to_datetime(ev["filed"]).dt.normalize()
    last = pd.Timestamp(known_through) if known_through is not None else ev["filed"].max()
    k = np.searchsorted(sessions, np.datetime64(last, "ns"), side="right")
    known = (naive.to_numpy() <= sessions[k]) if k < len(sessions) else np.ones(len(out), bool)
    pos = np.searchsorted(sessions, ev["filed"].to_numpy(dtype="datetime64[ns]"), side="right")
    keep = pos < len(sessions)
    ev = ev[keep].copy()
    ev["eff"] = sessions[pos[keep]]
    agg = ev.groupby(["symbol", "eff"])[["buy_shares", "sell_shares", "n_buyers", "n_sellers"]].sum().reset_index()
    key = pd.DataFrame({"symbol": out["symbol"].to_numpy(), "eff": naive.to_numpy()}, index=out.index)
    merged = key.merge(agg, on=["symbol", "eff"], how="left")
    covered = out["symbol"].isin(set(events["symbol"])).to_numpy()
    def col(c):
        v = merged[c].to_numpy(dtype=float)
        v = np.where(np.isnan(v) & covered, 0.0, v)
        return np.where(covered & known, v, np.nan)
    out["insider_buys"] = col("n_buyers")
    out["insider_sells"] = col("n_sellers")
    net = col("buy_shares") - col("sell_shares")
    out["insider_net_frac"] = net / _shares(out)
    out.loc[~np.isfinite(out["insider_net_frac"]), "insider_net_frac"] = np.nan
    return out


def attach_short_interest(df: pd.DataFrame, si: pd.DataFrame) -> pd.DataFrame:
    """FINRA consolidated short interest (data/fetch_finra_short_interest.py).

    short_ratio    latest public short interest / shares outstanding
    days_to_cover  FINRA's short interest / average daily volume

    A figure settled on session s is public only SHORT_INTEREST_LAG sessions
    later (usable ordinal = s + LAG) and is carried forward at most
    SHORT_MAX_AGE sessions; before a symbol's first figure both are NaN."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ord_of = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_ord"] = ord_of.reindex(naive.to_numpy()).to_numpy()
    s = si[["symbol", "settlement_date", "short_interest", "days_to_cover"]].copy()
    s["settlement_date"] = pd.to_datetime(s["settlement_date"]).dt.normalize()
    a = np.searchsorted(sessions, s["settlement_date"].to_numpy(dtype="datetime64[ns]"), side="left")
    keep = a < len(sessions)
    s = s[keep].copy()
    s["eff_ord"] = a[keep] + SHORT_INTEREST_LAG
    s = s.sort_values("eff_ord").drop_duplicates(["symbol", "eff_ord"], keep="last")
    left = out[["symbol", "_ord"]].reset_index().sort_values("_ord")
    merged = pd.merge_asof(left, s[["symbol", "eff_ord", "short_interest", "days_to_cover"]].sort_values("eff_ord"),
                           left_on="_ord", right_on="eff_ord", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    age = (merged["_ord"] - merged["eff_ord"]).astype(float)
    fresh = age <= SHORT_MAX_AGE
    short = merged["short_interest"].astype(float).where(fresh).reindex(out.index).to_numpy()
    out["short_ratio"] = short / _shares(out)
    out["days_to_cover"] = merged["days_to_cover"].astype(float).where(fresh).reindex(out.index).to_numpy()
    for c in ("short_ratio", "days_to_cover"):
        out.loc[~np.isfinite(out[c]), c] = np.nan
    return out.drop(columns=["_ord"])


def attach(df: pd.DataFrame, source: Path = EDGAR, filings: pd.DataFrame | None = None,
           news_source: Path = GDELT, news: pd.DataFrame | None = None,
           form4_source: Path = FORM4, form4: pd.DataFrame | None = None,
           short_source: Path = SHORT, short: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach every auxiliary field whose source is available; a field whose
    source is missing is simply absent (the DSL then refuses it). Order
    matters: the EDGAR share count feeds insider_net_frac and short_ratio."""
    out = df
    if filings is None and Path(source).exists():
        filings = pd.read_parquet(source, columns=["symbol", "filed", "shares_out"])
    if filings is not None:
        out = attach_filings(out, filings)
    if news is None and Path(news_source).exists():
        news = pd.read_parquet(news_source, columns=["symbol", "date", "gdelt_tone", "gdelt_articles"])
    if news is not None:
        out = attach_news(out, news)
    if form4 is None and Path(form4_source).exists():
        form4 = pd.read_parquet(form4_source)
    if form4 is not None:
        out = attach_form4(out, form4)
    if short is None and Path(short_source).exists():
        short = pd.read_parquet(short_source)
    if short is not None:
        out = attach_short_interest(out, short)
    return out


def available(df: pd.DataFrame) -> list[str]:
    return [f for f in AUX_FIELDS if f in df.columns]
