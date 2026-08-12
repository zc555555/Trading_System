"""Two-layer linear risk model (market + sector) and return attribution.

Model, per stock i on day t:

    r_it = beta_i,t * r_mkt,t + F_s(i),t + eps_it

  - r_mkt   : SPY close-to-close return (market proxy).
  - beta_i,t: rolling OLS beta of stock i vs market, estimated on a
              trailing window and SHIFTED one day -- the beta used to
              attribute day t was known at the close of t-1 (ex-ante,
              consistent with the repo's no-look-ahead discipline).
  - F_s,t   : sector factor return = equal-weight mean of the market-
              residual (r_it - beta_i,t * r_mkt,t) over the sector's
              members, unit loading on own sector.  11 Yahoo sectors +
              an explicit "Unknown" bucket (no guessing).
  - eps_it  : idiosyncratic return -- the part attributable to picking
              THIS stock rather than its market/sector.

Portfolio attribution for daily net weights w_it (long positive, short
negative, share of equity):

    market    = (sum_i w_it * beta_i,t) * r_mkt,t
    sector    = sum_s (sum_{i in s} w_it) * F_s,t
    selection = sum_i w_it * eps_it
    market + sector + selection == sum_i w_it * r_it   (exact, by construction)

When the caller supplies the *actual* realized daily return series (which
may differ from sum w*r via execution timing and trading costs), the gap
is reported as ``residual`` and never hidden inside selection.

This module is descriptive analytics only: it places no orders and feeds
nothing back into signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "stocks.parquet"
SECTOR_MAP_PATH = DATA_DIR / "sector_map.csv"

MARKET_SYMBOL = "SPY"
BETA_WINDOW = 252
BETA_MIN_PERIODS = 126
BETA_FALLBACK = 1.0   # used before a stock has BETA_MIN_PERIODS of history


@dataclass
class RiskModel:
    returns: pd.DataFrame        # date x symbol close-to-close returns
    market: pd.Series            # date -> market return
    beta: pd.DataFrame           # date x symbol ex-ante rolling beta
    sector_of: dict              # symbol -> sector name
    sector_returns: pd.DataFrame  # date x sector factor return
    resid: pd.DataFrame = field(repr=False, default=None)  # date x symbol eps

    def sectors(self) -> list[str]:
        return list(self.sector_returns.columns)


def load_sector_map(path: Path = SECTOR_MAP_PATH) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["symbol"], df["sector"].fillna("Unknown")))


def build_risk_model(
    prices: pd.DataFrame | None = None,
    sector_of: dict | None = None,
    window: int = BETA_WINDOW,
    min_periods: int = BETA_MIN_PERIODS,
) -> RiskModel:
    """Estimate betas and sector factor returns from the close-price panel.

    ``prices``: long DataFrame with columns date/symbol/close (defaults to
    the repo's raw panel, which includes SPY).
    """
    if prices is None:
        prices = pd.read_parquet(PANEL_PATH, columns=["date", "symbol", "close"])
    if sector_of is None:
        sector_of = load_sector_map()

    close = prices.pivot_table(index="date", columns="symbol", values="close")
    returns = close.pct_change(fill_method=None)

    if MARKET_SYMBOL not in returns.columns:
        raise ValueError(f"market proxy {MARKET_SYMBOL} missing from panel")
    market = returns[MARKET_SYMBOL].rename("market")
    stock_rets = returns.drop(columns=[MARKET_SYMBOL])

    # Rolling beta: cov(r_i, r_m) / var(r_m), trailing window, then shift(1)
    # so the beta attributing day t uses data only through t-1.
    mkt_var = market.rolling(window, min_periods=min_periods).var()
    cov = stock_rets.rolling(window, min_periods=min_periods).cov(market)
    beta = cov.div(mkt_var, axis=0).shift(1)
    beta = beta.clip(-1.0, 3.0)          # tame degenerate small-sample fits
    beta = beta.where(stock_rets.notna())  # no beta where no return
    beta = beta.fillna(BETA_FALLBACK).where(stock_rets.notna())

    # Market residuals and equal-weight sector factor returns.
    resid_mkt = stock_rets.sub(beta.mul(market, axis=0))
    sectors = pd.Series({s: sector_of.get(s, "Unknown") for s in stock_rets.columns})
    sector_returns = resid_mkt.T.groupby(sectors).mean().T
    # Idiosyncratic return: strip the own-sector factor from the residual.
    sector_of_col = sectors.reindex(stock_rets.columns)
    own_sector_ret = pd.DataFrame(
        sector_returns.reindex(columns=sector_of_col.values).to_numpy(),
        index=sector_returns.index, columns=stock_rets.columns,
    )
    resid = resid_mkt - own_sector_ret

    return RiskModel(returns=stock_rets, market=market, beta=beta,
                     sector_of=dict(sectors), sector_returns=sector_returns,
                     resid=resid)


def decompose_portfolio(
    rm: RiskModel,
    weights: pd.DataFrame,
    actual_daily: pd.Series | None = None,
    daily_costs: pd.Series | None = None,
) -> pd.DataFrame:
    """Attribute a stream of daily portfolio weights through the risk model.

    ``weights``: long DataFrame with columns date/symbol/weight -- NET
    weight as share of equity on that date (short = negative).  Dates
    should be the sessions over which the return accrues (weight held
    during day t is paired with day t's returns).

    Returns a date-indexed frame with columns:
      market, sector, selection : the three model components
      model_total               : their sum == sum_i w * r  (gross, close-to-close)
      beta_exposure, net_weight, gross_weight : daily exposures
      actual, costs, residual   : when actual/costs series are supplied;
                                  residual = actual - (model_total - costs),
                                  i.e. execution-timing noise.
    """
    w = weights.pivot_table(index="date", columns="symbol", values="weight",
                            aggfunc="sum")
    common_dates = w.index.intersection(rm.returns.index)
    common_syms = w.columns.intersection(rm.returns.columns)
    dropped = [s for s in w.columns if s not in set(common_syms)]
    if dropped:
        print(f"  [attribution] {len(dropped)} symbols not in risk model, "
              f"dropped: {dropped[:8]}{'...' if len(dropped) > 8 else ''}")
    w = w.loc[common_dates, common_syms]

    rets = rm.returns.loc[common_dates, common_syms]
    beta = rm.beta.loc[common_dates, common_syms]
    resid = rm.resid.loc[common_dates, common_syms]
    market = rm.market.loc[common_dates]

    # Where a held symbol has no return that day (halt/missing bar), its
    # contribution is unobservable in the model; treat as zero for all
    # components so additivity is preserved.
    live = rets.notna() & w.notna()
    w = w.where(live, 0.0)

    beta_exposure = (w * beta.fillna(BETA_FALLBACK)).sum(axis=1)
    market_comp = beta_exposure * market

    sector_of_col = pd.Series({s: rm.sector_of.get(s, "Unknown") for s in common_syms})
    sector_w = w.T.groupby(sector_of_col).sum().T           # date x sector net weight
    sec_rets = rm.sector_returns.loc[common_dates, sector_w.columns]
    sector_comp = (sector_w * sec_rets).sum(axis=1)

    selection_comp = (w * resid.fillna(0.0)).sum(axis=1)

    out = pd.DataFrame({
        "market": market_comp,
        "sector": sector_comp,
        "selection": selection_comp,
    })
    out["model_total"] = out.sum(axis=1)
    out["beta_exposure"] = beta_exposure
    out["net_weight"] = w.sum(axis=1)
    out["gross_weight"] = w.abs().sum(axis=1)

    if actual_daily is not None:
        actual = actual_daily.reindex(common_dates).fillna(0.0)
        costs = (daily_costs.reindex(common_dates).fillna(0.0)
                 if daily_costs is not None else pd.Series(0.0, index=common_dates))
        out["actual"] = actual
        out["costs"] = -costs.abs()          # always a drag, sign made explicit
        out["residual"] = actual - (out["model_total"] + out["costs"])
    return out


def summarize_attribution(attr: pd.DataFrame, ann_factor: int = 252) -> dict:
    """Annualized contribution of each component + realized exposures."""
    comps = [c for c in ("market", "sector", "selection", "costs", "residual")
             if c in attr.columns]
    n = len(attr)
    if n == 0:
        return {"n_days": 0}
    out = {"n_days": int(n)}
    total_col = "actual" if "actual" in attr.columns else "model_total"
    total_ann = attr[total_col].mean() * ann_factor
    out["ann_return_total"] = float(total_ann)
    for c in comps:
        out[f"ann_return_{c}"] = float(attr[c].mean() * ann_factor)
        sd = attr[c].std()
        if sd and np.isfinite(sd) and sd > 0:
            # per-component information ratio (mean/std, annualized)
            out[f"ir_{c}"] = float(attr[c].mean() / sd * np.sqrt(ann_factor))
    out["avg_beta_exposure"] = float(attr["beta_exposure"].mean())
    out["avg_net_weight"] = float(attr["net_weight"].mean())
    out["avg_gross_weight"] = float(attr["gross_weight"].mean())
    # Share of portfolio variance explained by market+sector vs selection.
    systematic = attr["market"] + attr["sector"]
    model_total_var = attr["model_total"].var()
    if model_total_var and np.isfinite(model_total_var) and model_total_var > 0:
        out["var_share_systematic"] = float(systematic.var() / model_total_var)
    return out
