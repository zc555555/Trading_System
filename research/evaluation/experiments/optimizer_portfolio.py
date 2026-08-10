"""Constrained portfolio optimizer experiment (merged-B item 3).

Upgrades construction from heuristics (top-N, inverse-vol) to a daily
whole-book QP:

    maximize   alpha @ w  -  tc * ||w - w_prev||_1
    subject to w' Sigma w <= sigma_daily^2          (vol target)
               |beta @ w| <= BETA_MAX               (partial beta hedge)
               |w_i|      <= NAME_CAP
               sum|w|     <= GROSS_MAX

- alpha_i = IC_PRIOR * zscore(pred)_i * vol20_i / HORIZON  (Grinold
  scaling: converts the unitless blend into a daily expected return so
  the TURNOVER TERM IS IN THE SAME UNITS AS REAL COSTS -- tc is the
  measured 7.5bp per side, not a tuning knob).
- Sigma: Ledoit-Wolf shrunk covariance of trailing 120d daily returns,
  re-estimated weekly. beta: trailing 120d OLS vs equal-weight market.
- Turnover cost endogenizes the holding period: the optimizer trades
  only when expected alpha clears the cost hurdle.

Verdict: four evidence tiers + full-period portfolio metrics vs the
incumbent construction (top-10 inverse-vol staggered) rerun on the SAME
members-only panel (pitOFF) at the same 15bp round-trip cost.
IC_PRIOR is declared a hyperparameter, chosen on dev only.

Usage:
    python evaluation/experiments/optimizer_portfolio.py --max-days 60  # smoke
    python evaluation/experiments/optimizer_portfolio.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import portfolio_metrics  # noqa: E402
from evaluation.experiments.portfolio_construction import (  # noqa: E402
    simulate as incumbent_simulate)

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "stocks_with_time_windows.parquet"
HOLDOUT = "2025-07-01"

HORIZON = 20
IC_PRIOR = 0.03          # hyperparameter (declared; dev-only selection)
SIGMA_TARGET_ANN = 0.10  # portfolio vol target
BETA_MAX = 0.5           # partial hedge -- the untested middle ground
NAME_CAP = 0.05
GROSS_MAX = 1.0
TC_PER_SIDE = 0.00075    # 7.5bp per side = 15bp round trip (calibrated)
# The daily objective is myopic: comparing ONE day's alpha against the
# full entry cost would never trade. Decision-time cost is amortized
# over the signal horizon (a position justified by an H-day alpha pays
# its entry over ~H days); REALIZED pnl still charges the full cost.
DEC_TC = TC_PER_SIDE / HORIZON
COV_WINDOW = 120
COV_REFRESH = 5          # re-estimate Sigma weekly


def load_inputs():
    panel = pd.read_parquet(RESULTS / "oos_predictions_h20_pitOFF.parquet")
    symbols = sorted(panel['symbol'].unique())
    prices = pd.read_parquet(DATA, columns=['date', 'symbol', 'close'])
    prices = prices[prices['symbol'].isin(symbols)]
    close = prices.pivot(index='date', columns='symbol', values='close').sort_index()
    rets = close.pct_change()
    # per-date z-score of the blend, prediction panel in wide form
    panel['z'] = (panel.groupby('date')['pred']
                  .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)))
    zwide = panel.pivot(index='date', columns='symbol', values='z').sort_index()
    vol20 = rets.rolling(20, min_periods=10).std()
    return close, rets, zwide, vol20


def solve_day(alpha, Sigma, beta, w_prev):
    import cvxpy as cp
    n = len(alpha)
    w = cp.Variable(n)
    sigma_daily = SIGMA_TARGET_ANN / np.sqrt(252)
    objective = cp.Maximize(alpha @ w - DEC_TC * cp.norm1(w - w_prev))
    cons = [
        cp.quad_form(w, cp.psd_wrap(Sigma)) <= sigma_daily ** 2,
        # absolute net-beta bound (DCP-safe: convex <= constant); with
        # gross <= 1 this caps portfolio beta at BETA_MAX
        cp.abs(beta @ w) <= BETA_MAX,
        cp.norm_inf(w) <= NAME_CAP,
        cp.norm1(w) <= GROSS_MAX,
    ]
    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return w_prev
        return np.asarray(w.value).ravel()
    except Exception:
        return w_prev


def run_optimizer(close, rets, zwide, vol20, max_days=None):
    from sklearn.covariance import LedoitWolf
    dates = [d for d in zwide.index if d in rets.index]
    if max_days:
        dates = dates[:max_days]
    symbols = list(zwide.columns)
    n = len(symbols)
    w_prev = np.zeros(n)
    daily_ret, daily_to = {}, {}
    Sigma = None
    t0 = time.time()

    ret_idx = rets.index.get_indexer(pd.Index(dates))
    for k, date in enumerate(dates):
        i = ret_idx[k]
        if i < COV_WINDOW + 5:
            continue
        window = rets.iloc[i - COV_WINDOW:i][symbols]
        live = window.columns[window.notna().sum() >= COV_WINDOW * 0.8]
        if len(live) < 50:
            continue
        if Sigma is None or k % COV_REFRESH == 0:
            X = window[live].fillna(0.0).to_numpy()
            lw = LedoitWolf().fit(X)
            Sigma = lw.covariance_
            mkt = window[live].mean(axis=1)
            cov_m = window[live].apply(lambda c: c.cov(mkt))
            beta = (cov_m / (mkt.var() or 1e-8)).clip(-1, 3).to_numpy()
            live_cached = list(live)
        # align alpha/w_prev to the cached live set
        z = zwide.loc[date, live_cached].to_numpy()
        v = vol20.iloc[i - 1][live_cached].to_numpy()
        alpha = IC_PRIOR * np.nan_to_num(z) * np.nan_to_num(v, nan=0.02) / HORIZON
        prev_map = dict(zip(symbols, w_prev))
        wp = np.array([prev_map.get(s, 0.0) for s in live_cached])

        w = solve_day(alpha, Sigma, beta, wp)

        nxt = rets.iloc[i + 1] if i + 1 < len(rets) else None
        if nxt is not None:
            r = np.nan_to_num(nxt[live_cached].to_numpy())
            turnover = np.abs(w - wp).sum()
            daily_ret[rets.index[i + 1]] = float(w @ r - TC_PER_SIDE * turnover)
            daily_to[rets.index[i + 1]] = float(turnover)
        new_map = dict(zip(live_cached, w))
        w_prev = np.array([new_map.get(s, 0.0) for s in symbols])

        if k % 250 == 0:
            print(f"  [{k}/{len(dates)}] {pd.Timestamp(date).date()} "
                  f"({(time.time() - t0):.0f}s)")
    return pd.Series(daily_ret).sort_index(), pd.Series(daily_to).sort_index()


def report(name, daily):
    idx = pd.DatetimeIndex(daily.index)
    cut = pd.Timestamp(HOLDOUT)
    if idx.tz is not None:
        cut = cut.tz_localize(idx.tz)
    print(f"\n=== {name} ===")
    out = {}
    for seg, series in (("dev", daily[idx < cut]), ("holdout", daily[idx >= cut]),
                        ("all", daily)):
        m = portfolio_metrics(series)
        out[seg] = m
        if m:
            print(f"  {seg:<8} sharpe={m['sharpe']:>6.2f}  ann={m['ann_return']:>+7.1%}  "
                  f"vol={m['ann_vol']:>6.1%}  maxDD={m['max_drawdown']:>+7.1%}  "
                  f"days={m['n_days']}")
    return out


def main(max_days=None):
    print("loading inputs (members-only h20 panel)...")
    close, rets, zwide, vol20 = load_inputs()
    print(f"panel: {zwide.shape[0]} signal days x {zwide.shape[1]} symbols")

    print("\nrunning optimizer book...")
    opt_ret, opt_to = run_optimizer(close, rets, zwide, vol20, max_days=max_days)
    print(f"avg daily turnover: {opt_to.mean():.1%} (annualized ~{opt_to.mean()*252:.0f}x gross)")
    r_opt = report("optimizer (vol-target 10%, beta<=0.5, tc-endogenous)", opt_ret)

    if not max_days:
        print("\nrunning incumbent construction on the same panel...")
        panel = pd.read_parquet(RESULTS / "oos_predictions_h20_pitOFF.parquet")
        prices = pd.read_parquet(DATA, columns=['date', 'symbol', 'open', 'close'])
        g = prices.sort_values(['symbol', 'date']).groupby('symbol')
        prices['vol20'] = (g['close'].pct_change()
                           .groupby(prices['symbol']).transform(
                               lambda s: s.rolling(20, min_periods=10).std())
                           * np.sqrt(252))
        panel = panel.merge(prices[['date', 'symbol', 'vol20']],
                            on=['date', 'symbol'], how='left')
        sessions = np.sort(prices['date'].unique())
        wide = {'sessions': list(sessions),
                'idx': {d: i for i, d in enumerate(sessions)},
                'open': prices.set_index(['date', 'symbol'])['open'].to_dict(),
                'close': prices.set_index(['date', 'symbol'])['close'].to_dict()}
        import evaluation.experiments.portfolio_construction as pc
        pc.HOLD_DAYS = 20
        inc_ret = incumbent_simulate(panel, wide, top_n=10, inv_vol=True,
                                     neutral=False, cost_bp=15.0)
        report("incumbent (top-10 ivol staggered, 15bp)", inc_ret)

        pd.DataFrame({'optimizer': opt_ret, 'turnover': opt_to}).to_csv(
            RESULTS / "optimizer_daily.csv")
        print(f"\nsaved: {RESULTS / 'optimizer_daily.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    main(max_days=args.max_days)
