"""Point-in-time fundamental features from the EDGAR extract (wave 2).

Availability discipline: a quarterly value becomes usable the trading
day AFTER its filing date; TTM aggregates become usable only when ALL
four constituent quarters have been filed (availability = max of the
four filing dates). Features are as-of merged onto the trading grid --
no value is ever visible before the market saw the filing.

Factor groups built here (each a wave-2 candidate):
    value          ep_ttm  = TTM net income / market cap
                   bp      = latest equity / market cap
    profitability  gpa_ttm = TTM gross profit / latest assets
                   roe_ttm = TTM net income / latest equity
    investment     asset_growth = assets / assets four quarters ago - 1
    accruals       accruals_ttm = (TTM NI - TTM CFO) / latest assets

Market cap = PIT shares outstanding x same-day close. Financial-sector
filers often lack gross profit / COGS -- coverage gaps stay NaN and the
training core treats them as absent (documented residual).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FUND = Path(__file__).parent.parent / "data" / "fundamentals.parquet"


def _avail_series(g: pd.DataFrame, ttm: bool) -> pd.DataFrame:
    """Per (symbol, concept): availability-dated values.

    ttm=True  -> rolling 4-quarter sum, available at max(filed) of the 4
    ttm=False -> latest instant value, available at its filed date
    """
    g = g.sort_values("end").drop_duplicates("end", keep="first")
    if ttm:
        if len(g) < 4:
            return pd.DataFrame()
        vals = g["val"].rolling(4).sum()
        avail = g["filed"].rolling(4, min_periods=4).max()
        out = pd.DataFrame({"avail": avail, "value": vals,
                            "end": g["end"]}).dropna()
    else:
        out = pd.DataFrame({"avail": g["filed"], "value": g["val"],
                            "end": g["end"]})
    return out


def _asof_onto(df: pd.DataFrame, series: pd.DataFrame, col: str,
               max_age_days: int = 400) -> pd.Series:
    """As-of merge availability-dated values onto (symbol, date) rows.
    Values older than max_age_days are considered stale -> NaN."""
    if series.empty:
        return pd.Series(np.nan, index=df.index)
    left = (df[["symbol", "date"]].reset_index()
            .assign(dnaive=lambda d: d["date"].dt.tz_localize(None))
            .sort_values("dnaive"))
    right = (series.rename(columns={"avail": "dnaive"})
             .assign(dnaive=lambda d: pd.to_datetime(d["dnaive"]))
             .sort_values("dnaive"))
    merged = pd.merge_asof(left, right, on="dnaive", by="symbol",
                           direction="backward")
    age = (merged["dnaive"] - pd.to_datetime(merged["end"])).dt.days
    merged.loc[age > max_age_days, "value"] = np.nan
    return merged.set_index("index")["value"].rename(col)


def build_fundamental_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Append PIT fundamental features; returns (df, factor feature map)."""
    fund = pd.read_parquet(FUND)
    fund["filed"] = pd.to_datetime(fund["filed"])
    fund["end"] = pd.to_datetime(fund["end"])
    fund = fund[fund["symbol"].isin(df["symbol"].unique())]

    def concept(name, ttm):
        sub = fund[fund["concept"] == name]
        parts = [(_avail_series(s, ttm).assign(symbol=sym))
                 for sym, s in sub.groupby("symbol")]
        parts = [p for p in parts if len(p)]
        return (pd.concat(parts, ignore_index=True)
                if parts else pd.DataFrame())

    ni_ttm = concept("net_income", ttm=True)
    cfo_ttm = concept("op_cashflow", ttm=True)
    gp_ttm = concept("gross_profit", ttm=True)
    rev_ttm = concept("revenue", ttm=True)
    cogs_ttm = concept("cogs", ttm=True)
    assets = concept("assets", ttm=False)
    equity = concept("equity", ttm=False)
    shares = concept("shares_out", ttm=False)

    # assets four quarters ago (for asset growth): shift the instant series
    ag_parts = []
    for sym, s in fund[fund["concept"] == "assets"].groupby("symbol"):
        s = s.sort_values("end").drop_duplicates("end", keep="first")
        if len(s) < 5:
            continue
        ag_parts.append(pd.DataFrame({
            "avail": s["filed"].values, "end": s["end"].values,
            "value": s["val"].values / s["val"].shift(4).values - 1.0,
            "symbol": sym}))
    asset_growth = (pd.concat(ag_parts, ignore_index=True).dropna()
                    if ag_parts else pd.DataFrame())

    df = df.copy()
    df["_ni"] = _asof_onto(df, ni_ttm, "_ni")
    df["_cfo"] = _asof_onto(df, cfo_ttm, "_cfo")
    df["_gp"] = _asof_onto(df, gp_ttm, "_gp")
    df["_rev"] = _asof_onto(df, rev_ttm, "_rev")
    df["_cogs"] = _asof_onto(df, cogs_ttm, "_cogs")
    df["_assets"] = _asof_onto(df, assets, "_assets")
    df["_equity"] = _asof_onto(df, equity, "_equity")
    df["_shares"] = _asof_onto(df, shares, "_shares")
    df["_agrow"] = _asof_onto(df, asset_growth, "_agrow")

    mktcap = (df["_shares"] * df["close"]).replace(0, np.nan)
    gp = df["_gp"].fillna(df["_rev"] - df["_cogs"])

    df["ep_ttm"] = df["_ni"] / mktcap
    df["bp"] = df["_equity"] / mktcap
    df["gpa_ttm"] = gp / df["_assets"].replace(0, np.nan)
    df["roe_ttm"] = df["_ni"] / df["_equity"].replace(0, np.nan)
    df["asset_growth"] = df["_agrow"]
    df["accruals_ttm"] = (df["_ni"] - df["_cfo"]) / df["_assets"].replace(0, np.nan)

    feats = {
        "value": ["ep_ttm", "bp"],
        "profitability": ["gpa_ttm", "roe_ttm"],
        "investment": ["asset_growth"],
        "accruals": ["accruals_ttm"],
    }
    for cols in feats.values():
        for c in cols:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-10, 10)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df, feats
