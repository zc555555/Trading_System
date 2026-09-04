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
XBRL = DATA / "xbrl_fundamentals.parquet"
REGSHO = DATA / "regsho_short_volume.parquet"
FORM13F = DATA / "form13f_quarterly.parquet"
INST_MAX_AGE = 70             # sessions a 13F quarter is carried (one quarter plus slack)
# every path keyword attach() accepts; tests pass a nonexistent path for each to get a bare panel
SOURCE_KWARGS = ("source", "news_source", "form4_source", "short_source", "xbrl_source", "regsho_source", "form13f_source")
FUNDAMENTAL_FIELDS = ("book_to_market", "earnings_yield", "sales_to_price", "gross_profitability", "roe",
                      "asset_growth", "accruals", "leverage", "cash_to_assets", "rd_to_sales",
                      "capex_to_assets", "op_margin")
AUX_FIELDS = ("marketcap", "turnover", "filing_days", "news_tone", "news_articles",
              "insider_buys", "insider_sells", "insider_net_frac", "short_ratio", "days_to_cover") + FUNDAMENTAL_FIELDS \
             + ("short_vol_ratio", "inst_own", "inst_holders", "inst_top5")
FUNDAMENTALS_MAX_AGE = 300    # sessions without any filing -> the fundamentals are unknown
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


XBRL_COLS = ["revenue_ttm", "gp_ttm", "ni_ttm", "ocf_ttm", "capex_ttm", "rd_ttm", "opinc_ttm",
             "assets", "equity", "liabilities", "ltdebt", "cash", "assets_1y"]


def attach_fundamentals(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time accounting ratios (data/build_xbrl_fundamentals.py).

    A filing on session a is usable from a + 1 (same convention as the
    share count); the latest filing is carried at most FUNDAMENTALS_MAX_AGE
    sessions. Ratios (all NaN when a component is unknown or the
    denominator is not positive):
      book_to_market      equity / market cap
      earnings_yield      net income (TTM) / market cap
      sales_to_price      revenue (TTM) / market cap
      gross_profitability gross profit (TTM) / assets        (Novy-Marx)
      roe                 net income (TTM) / equity
      asset_growth        assets / assets one year earlier - 1 (Cooper et al.)
      accruals            (net income - operating cash flow) / assets (Sloan)
      leverage            long-term debt / assets; 0 when the company reports
                          no long-term debt tag but does report assets
      cash_to_assets      cash / assets
      rd_to_sales         R&D (TTM) / revenue; 0 when no R&D is reported
      capex_to_assets     capex (TTM) / assets
      op_margin           operating income (TTM) / revenue"""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ord_of = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_ord"] = ord_of.reindex(naive.to_numpy()).to_numpy()
    f = fundamentals[["symbol", "filed"] + [c for c in XBRL_COLS if c in fundamentals.columns]].copy()
    for c in XBRL_COLS:
        if c not in f.columns:
            f[c] = np.nan
    f["filed"] = pd.to_datetime(f["filed"]).dt.normalize()
    a = np.searchsorted(sessions, f["filed"].to_numpy(dtype="datetime64[ns]"), side="left")
    keep = a < len(sessions)
    f = f[keep].copy()
    f["eff_ord"] = a[keep] + 1
    f = f.sort_values("eff_ord").drop_duplicates(["symbol", "eff_ord"], keep="last")
    left = out[["symbol", "_ord"]].reset_index().sort_values("_ord")
    merged = pd.merge_asof(left, f[["symbol", "eff_ord"] + XBRL_COLS].sort_values("eff_ord"),
                           left_on="_ord", right_on="eff_ord", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    fresh = (merged["_ord"] - merged["eff_ord"]).astype(float) <= FUNDAMENTALS_MAX_AGE
    v = {c: merged[c].astype(float).where(fresh).reindex(out.index).to_numpy() for c in XBRL_COLS}
    mcap = (out["marketcap"].to_numpy(dtype=float) * 1e6) if "marketcap" in out.columns else np.full(len(out), np.nan)

    def ratio(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den
        return np.where(np.isfinite(r) & (den > 0), r, np.nan)

    assets_known = np.isfinite(v["assets"]) & (v["assets"] > 0)
    ltdebt = np.where(np.isnan(v["ltdebt"]) & assets_known, 0.0, v["ltdebt"])
    rd = np.where(np.isnan(v["rd_ttm"]) & np.isfinite(v["revenue_ttm"]), 0.0, v["rd_ttm"])
    out["book_to_market"] = ratio(v["equity"], mcap)
    out["earnings_yield"] = ratio(v["ni_ttm"], mcap)
    out["sales_to_price"] = ratio(v["revenue_ttm"], mcap)
    out["gross_profitability"] = ratio(v["gp_ttm"], v["assets"])
    out["roe"] = ratio(v["ni_ttm"], v["equity"])
    out["asset_growth"] = ratio(v["assets"], v["assets_1y"]) - 1.0
    out["accruals"] = ratio(v["ni_ttm"] - v["ocf_ttm"], v["assets"])
    out["leverage"] = ratio(ltdebt, v["assets"])
    out["cash_to_assets"] = ratio(v["cash"], v["assets"])
    out["rd_to_sales"] = ratio(rd, v["revenue_ttm"])
    out["capex_to_assets"] = ratio(v["capex_ttm"], v["assets"])
    out["op_margin"] = ratio(v["opinc_ttm"], v["revenue_ttm"])
    for c in FUNDAMENTAL_FIELDS:
        out.loc[~np.isfinite(out[c]), c] = np.nan
    return out.drop(columns=["_ord"])


def attach_short_volume(df: pd.DataFrame, sv: pd.DataFrame, known_through=None) -> pd.DataFrame:
    """FINRA Reg SHO daily short sale volume (data/fetch_regsho_short_volume.py).

    short_vol_ratio  short volume / total FINRA-reported volume of the LAST
                     session (a session's file is published that evening, so
                     it is usable from the next session); NaN when the symbol
                     has no row that day, is not in the source, or the date is
                     after the source's last covered session."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    s = sv[["symbol", "date", "short_ratio_day"]].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    pos = np.searchsorted(sessions, s["date"].to_numpy(dtype="datetime64[ns]"), side="right")
    keep = pos < len(sessions)
    s = s[keep].copy()
    s["eff"] = sessions[pos[keep]]
    s = s.sort_values(["symbol", "eff", "date"]).drop_duplicates(["symbol", "eff"], keep="last")
    key = pd.DataFrame({"symbol": out["symbol"].to_numpy(), "eff": naive.to_numpy()}, index=out.index)
    merged = key.merge(s[["symbol", "eff", "short_ratio_day"]], on=["symbol", "eff"], how="left")
    last = pd.Timestamp(known_through) if known_through is not None else s["date"].max()
    k = np.searchsorted(sessions, np.datetime64(last, "ns"), side="right")
    known = (naive.to_numpy() <= sessions[k]) if k < len(sessions) else np.ones(len(out), bool)
    v = merged["short_ratio_day"].to_numpy(dtype=float)
    out["short_vol_ratio"] = np.where(known, v, np.nan)
    return out


def attach_form13f(df: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Institutional ownership from Form 13F (data/build_form13f_fields.py).

    inst_own      13F-reported shares / shares outstanding (EDGAR)
    inst_holders  number of 13F managers holding the stock
    inst_top5     share of institutional holdings held by the five largest

    A quarter P is usable from the first session strictly after its
    `usable_from` date (P + 45 days, the filing deadline) and is carried at
    most INST_MAX_AGE sessions; NaN before a symbol's first quarter."""
    out = df.copy()
    naive = _naive_dates(out["date"])
    sessions = np.sort(naive.unique())
    ord_of = pd.Series(np.arange(len(sessions)), index=sessions)
    out["_ord"] = ord_of.reindex(naive.to_numpy()).to_numpy()
    f = q[["symbol", "usable_from", "inst_shares", "inst_holders", "top5_share"]].copy()
    f["usable_from"] = pd.to_datetime(f["usable_from"]).dt.normalize()
    a = np.searchsorted(sessions, f["usable_from"].to_numpy(dtype="datetime64[ns]"), side="right")
    keep = a < len(sessions)
    f = f[keep].copy()
    f["eff_ord"] = a[keep]
    f = f.sort_values("eff_ord").drop_duplicates(["symbol", "eff_ord"], keep="last")
    left = out[["symbol", "_ord"]].reset_index().sort_values("_ord")
    merged = pd.merge_asof(left, f[["symbol", "eff_ord", "inst_shares", "inst_holders", "top5_share"]].sort_values("eff_ord"),
                           left_on="_ord", right_on="eff_ord", by="symbol", direction="backward")
    merged = merged.set_index("index").sort_index()
    fresh = (merged["_ord"] - merged["eff_ord"]).astype(float) <= INST_MAX_AGE
    shares = merged["inst_shares"].astype(float).where(fresh).reindex(out.index).to_numpy()
    out["inst_own"] = shares / _shares(out)
    out["inst_holders"] = merged["inst_holders"].astype(float).where(fresh).reindex(out.index).to_numpy()
    out["inst_top5"] = merged["top5_share"].astype(float).where(fresh).reindex(out.index).to_numpy()
    for c in ("inst_own", "inst_holders", "inst_top5"):
        out.loc[~np.isfinite(out[c]), c] = np.nan
    return out.drop(columns=["_ord"])


def attach(df: pd.DataFrame, source: Path = EDGAR, filings: pd.DataFrame | None = None,
           news_source: Path = GDELT, news: pd.DataFrame | None = None,
           form4_source: Path = FORM4, form4: pd.DataFrame | None = None,
           short_source: Path = SHORT, short: pd.DataFrame | None = None,
           xbrl_source: Path = XBRL, fundamentals: pd.DataFrame | None = None,
           regsho_source: Path = REGSHO, short_volume: pd.DataFrame | None = None,
           form13f_source: Path = FORM13F, form13f: pd.DataFrame | None = None) -> pd.DataFrame:
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
    if fundamentals is None and Path(xbrl_source).exists():
        fundamentals = pd.read_parquet(xbrl_source)
    if fundamentals is not None:
        out = attach_fundamentals(out, fundamentals)
    if short_volume is None and Path(regsho_source).exists():
        short_volume = pd.read_parquet(regsho_source, columns=["symbol", "date", "short_ratio_day"])
    if short_volume is not None:
        out = attach_short_volume(out, short_volume)
    if form13f is None and Path(form13f_source).exists():
        form13f = pd.read_parquet(form13f_source)
    if form13f is not None:
        out = attach_form13f(out, form13f)
    return out


def available(df: pd.DataFrame) -> list[str]:
    return [f for f in AUX_FIELDS if f in df.columns]
