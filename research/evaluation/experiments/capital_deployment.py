"""Hypothesis capital_deployment_rate (registered 2026-09-15): the live book
deploys only ~53% of equity. Three levers on the ledger engine (production
selection, caps on, brackets OFF = the live book since 2026-09-15):

  (a) capital_per_tranche_pct above 100 / hold_days (7.5%, 10%), the
      no-leverage guard as the binding cap;
  (b) fractional shares for LONG legs (Alpaca supports them; shorts whole);
  (c) a larger account ($1M) so no name is priced above its allocation.

Rule (ledger): adopted only if dev Sharpe is not lower than the baseline by
more than 0.05, max drawdown not deeper by more than 3 points, and holdout
not moving the other way by more than 0.05.

Usage:  python evaluation/experiments/capital_deployment.py
Output: results/capital_deployment_levers_h20_surv_ledger.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation import ledger_sim as ls  # noqa: E402
from evaluation.experiments import portfolio_construction as pc  # noqa: E402
from evaluation.experiments import production_replay as rp  # noqa: E402
from evaluation.metrics import portfolio_metrics  # noqa: E402
from strategy import selection as sel  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "stocks_with_time_windows_surv.parquet"

VARIANTS = {
    "baseline_5pct_100k": dict(),
    "tranche_7.5pct": dict(capital_per_tranche_pct=7.5),
    "tranche_10pct": dict(capital_per_tranche_pct=10.0),
    "fractional_longs": dict(fractional_longs=True),
    "equity_1M": dict(initial_equity=1_000_000.0),
    "fractional_longs_1M": dict(fractional_longs=True, initial_equity=1_000_000.0),
}


def main(horizon: int = 20) -> dict:
    pc.HOLD_DAYS = horizon
    pc.DATA = DATA
    panel, _ = pc.load_inputs(horizon, "surv")
    prices = pd.read_parquet(DATA, columns=["date", "symbol", "open", "high", "low", "close", "volatility_20d"])
    params = sel.SelectionParams.production()
    v = panel.copy()
    v["pred"] = rp.variant_predictions(panel, prices, params).to_numpy()
    v = v.merge(prices[["date", "symbol", "volatility_20d"]], on=["date", "symbol"], how="left")   # production sizing input
    books = ls.books_from_predictions(v, "pred", params)
    lo, hi = pd.Timestamp(min(books)), pd.Timestamp(panel["date"].max())
    cut = pd.Timestamp(pc.HOLDOUT)
    out = {}
    print(f"{'variant':<22}{'seg':<9}{'sharpe':>7}{'ann':>8}{'maxDD':>8}{'gross%':>8}{'blocked':>10}")
    for name, over in VARIANTS.items():
        lp = ls.LedgerParams(hold_days=horizon, cost_bp=15.0, apply_caps=True, brackets=False, **over)
        res = ls.simulate_ledger(books, prices, lp, collect=True)
        ridx = pd.DatetimeIndex(res.returns.index)
        lo_, hi_ = (lo.tz_localize(None), hi.tz_localize(None)) if (ridx.tz is None and lo.tz is not None) else (lo, hi)
        daily = res.returns[(ridx >= lo_) & (ridx <= hi_)]
        w = res.weights[(pd.DatetimeIndex(res.weights["date"]) >= lo_) & (pd.DatetimeIndex(res.weights["date"]) <= hi_)]
        gross = w.assign(a=w["weight"].abs()).groupby("date")["a"].sum()
        rec = {"params": over, "blocked": res.blocked, "exits": res.stats["exits"], "avg_gross": float(gross.mean())}
        idx = pd.DatetimeIndex(daily.index)
        c = cut.tz_localize(idx.tz) if idx.tz is not None else cut
        for seg, series in (("dev", daily[idx < c]), ("holdout", daily[idx >= c]), ("all", daily)):
            m = portfolio_metrics(series)
            rec[seg] = {k: m[k] for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days")}
            print(f"{name:<22}{seg:<9}{m['sharpe']:>7.2f}{m['ann_return']:>8.1%}{m['max_drawdown']:>8.1%}"
                  f"{rec['avg_gross']:>8.0%}{sum(res.blocked.values()):>10}")
        out[name] = rec
    path = RESULTS / f"capital_deployment_levers_h{horizon}_surv_ledger.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"saved: {path}")
    return out


if __name__ == "__main__":
    main()
