"""Unit tests for the two-layer risk model (research/evaluation/risk_model.py).

Pins the two properties the attribution's credibility rests on:
  1. rolling betas recover known betas on synthetic data (ex-ante, shifted);
  2. market + sector + selection == sum(w * r) exactly (additivity), and
     the actual-vs-model gap lands in `residual`, never inside selection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation.risk_model import (  # noqa: E402
    build_risk_model, decompose_portfolio,
)

RNG = np.random.default_rng(7)


def make_synthetic(n_days=400, betas=None):
    """Price panel with known market betas and two sectors."""
    betas = betas or {"AAA": 0.5, "BBB": 1.5, "CCC": 1.0, "DDD": 1.0}
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    r_mkt = RNG.normal(0.0004, 0.01, n_days)
    rets = {"SPY": r_mkt}
    for sym, b in betas.items():
        rets[sym] = b * r_mkt + RNG.normal(0, 0.004, n_days)
    rows = []
    for sym, r in rets.items():
        px = 100 * np.cumprod(1 + np.asarray(r))
        rows.append(pd.DataFrame({"date": dates, "symbol": sym, "close": px}))
    prices = pd.concat(rows, ignore_index=True)
    sector_of = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy"}
    return prices, sector_of, betas


def test_beta_recovery():
    prices, sector_of, betas = make_synthetic()
    rm = build_risk_model(prices=prices, sector_of=sector_of,
                          window=252, min_periods=126)
    last = rm.beta.iloc[-1]
    for sym, true_beta in betas.items():
        assert abs(last[sym] - true_beta) < 0.25, \
            f"{sym}: estimated {last[sym]:.2f} vs true {true_beta}"


def test_decomposition_additivity():
    prices, sector_of, _ = make_synthetic()
    rm = build_risk_model(prices=prices, sector_of=sector_of,
                          window=252, min_periods=126)
    dates = rm.returns.index[300:340]
    weights = pd.DataFrame(
        [{"date": d, "symbol": s, "weight": w}
         for d in dates
         for s, w in (("AAA", 0.30), ("BBB", 0.25), ("CCC", -0.02), ("DDD", 0.15))]
    )
    attr = decompose_portfolio(rm, weights)

    # additivity: the three components must sum to sum(w * r) exactly
    w_wide = weights.pivot(index="date", columns="symbol", values="weight")
    direct = (w_wide * rm.returns.loc[dates, w_wide.columns]).sum(axis=1)
    parts = attr["market"] + attr["sector"] + attr["selection"]
    assert np.allclose(parts, direct, atol=1e-12)
    assert np.allclose(attr["model_total"], direct, atol=1e-12)


def test_actual_gap_goes_to_residual():
    prices, sector_of, _ = make_synthetic()
    rm = build_risk_model(prices=prices, sector_of=sector_of,
                          window=252, min_periods=126)
    dates = rm.returns.index[300:320]
    weights = pd.DataFrame(
        [{"date": d, "symbol": "AAA", "weight": 0.5} for d in dates]
    )
    # actual return differs from the model's close-to-close by 10bp/day
    actual = pd.Series(0.001, index=dates) + \
        0.5 * rm.returns.loc[dates, "AAA"]
    costs = pd.Series(0.0002, index=dates)
    attr = decompose_portfolio(rm, weights, actual_daily=actual,
                               daily_costs=costs)
    # identity: actual == model_total + costs + residual (costs stored negative)
    recon = attr["model_total"] + attr["costs"] + attr["residual"]
    assert np.allclose(recon, attr["actual"], atol=1e-12)
    # selection must NOT absorb the gap: it only contains idiosyncratic
    # close-to-close returns, which for a single stock inside its own
    # 2-stock sector bucket is well-defined and independent of `actual`.
    attr_no_actual = decompose_portfolio(rm, weights)
    assert np.allclose(attr["selection"], attr_no_actual["selection"], atol=1e-15)


def test_shorts_attribute_with_negative_weight():
    prices, sector_of, _ = make_synthetic()
    rm = build_risk_model(prices=prices, sector_of=sector_of,
                          window=252, min_periods=126)
    dates = rm.returns.index[300:310]
    long_w = pd.DataFrame([{"date": d, "symbol": "BBB", "weight": 0.4}
                           for d in dates])
    short_w = pd.DataFrame([{"date": d, "symbol": "BBB", "weight": -0.4}
                            for d in dates])
    a_long = decompose_portfolio(rm, long_w)
    a_short = decompose_portfolio(rm, short_w)
    assert np.allclose(a_long["model_total"], -a_short["model_total"], atol=1e-12)
    assert np.allclose(a_long["beta_exposure"], -a_short["beta_exposure"], atol=1e-12)
