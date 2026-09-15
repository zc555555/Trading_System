"""Replay the PRODUCTION selection policy over the walk-forward predictions
(2026-09-14 review item 3: the evaluation must measure the strategy that
actually trades).

Until now the evaluation book took each date's blended prediction as is;
production smooths the last five sessions' scores and drops names that fail
the long-term trend filter before ranking. strategy/selection.py is the one
implementation of those steps, used by the nightly script and here.

Variants (all with the production construction: top-10, inverse-vol, raw
long/short, 15 bp):

  raw         the old evaluation book (no smoothing, no filter)
  smooth      5-day smoothing only
  filter      trend filter only
  production  smoothing + trend filter  (= the live book's selection)

For each: dev / holdout Sharpe, annual return, max drawdown, and the mean
daily rank IC of the variant's prediction against the label. The two
switch hypotheses are pre-registered in the ledger before this runs;
this script only produces the numbers.

Usage:
    python evaluation/experiments/production_replay.py --horizon 20 --panel surv \
        --data data/stocks_with_time_windows_surv.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation import ledger_sim as ls  # noqa: E402
from evaluation.experiments import portfolio_construction as pc  # noqa: E402
from evaluation.metrics import portfolio_metrics  # noqa: E402
from strategy import selection as sel  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
VARIANTS = {"raw": dict(smooth_days=1, trend_filter=False),
            "smooth": dict(trend_filter=False),
            "filter": dict(smooth_days=1),
            "production": {}}


def variant_predictions(panel: pd.DataFrame, prices: pd.DataFrame, params: sel.SelectionParams) -> pd.Series:
    """The prediction production would have ranked on each date under
    `params`: smoothed over sessions, NaN where the trend filter fails."""
    pred = sel.smooth_panel(panel[["date", "symbol", "pred"]], "pred", params)
    if params.trend_filter:
        metrics = sel.trend_metrics_panel(prices)
        m = panel[["date", "symbol"]].merge(metrics, on=["date", "symbol"], how="left")
        ok = sel.trend_mask(m, params).to_numpy()
        pred = pred.where(ok)
    return pred


def daily_rank_ic(panel: pd.DataFrame, pred_col: str, label: str) -> tuple[float, int]:
    ics = []
    for _, day in panel.dropna(subset=[pred_col, label]).groupby("date", sort=True):
        if len(day) >= 40:
            ics.append(day[pred_col].rank().corr(day[label].rank()))
    return (float(np.nanmean(ics)) if ics else float("nan")), len(ics)


def run(horizon: int, panel_tag: str, data_path: str | None, top_n: int = 10, cost_bp: float = 15.0,
        engine: str = "weights", apply_caps: bool = True, brackets: bool = True) -> dict:
    """engine='weights': the construction grid's fixed-weight simulator
    (daily rebalanced marks); engine='ledger': evaluation/ledger_sim.py --
    whole shares, tranche capital from equity, the no-debt / short caps,
    scheduled open-to-open exits and ATR brackets, i.e. the orchestrator."""
    pc.HOLD_DAYS = horizon
    if data_path:
        pc.DATA = Path(data_path)
    panel, wide = pc.load_inputs(horizon, panel_tag)
    prices = pd.read_parquet(pc.DATA, columns=["date", "symbol", "open", "high", "low", "close", "volatility_20d"])
    label = f"future_return_{horizon}d"
    base = sel.SelectionParams.production(top_n=top_n)
    lp = ls.LedgerParams(hold_days=horizon, capital_per_tranche_pct=100.0 / horizon, cost_bp=cost_bp,
                         apply_caps=apply_caps, brackets=brackets)
    out = {"horizon": horizon, "panel": panel_tag, "top_n": top_n, "cost_bp": cost_bp, "engine": engine,
           "ledger_params": lp.__dict__ if engine == "ledger" else None,
           "params": base.to_dict(), "holdout_start": pc.HOLDOUT, "variants": {}}
    cut = pd.Timestamp(pc.HOLDOUT)
    print(f"{'variant':<12}{'seg':<9}{'sharpe':>7}{'ann':>8}{'maxDD':>8}{'IC':>8}{'n_days':>8}")
    print("-" * 60)
    for name, over in VARIANTS.items():
        params = sel.SelectionParams.production(top_n=top_n, **over)
        v = panel[["date", "symbol", "open", "close", label, "vol20"]].copy()
        v["pred"] = variant_predictions(panel, prices, params).to_numpy()
        ic, n_ic = daily_rank_ic(v, "pred", label)
        rec = {"params": params.to_dict(), "ic_mean": ic, "ic_days": n_ic,
               "coverage": float(v["pred"].notna().mean())}
        if engine == "ledger":
            vv = v.merge(prices[["date", "symbol", "volatility_20d"]], on=["date", "symbol"], how="left")
            books = ls.books_from_predictions(vv, "pred", params)
            res = ls.simulate_ledger(books, prices, lp)
            # the price panel starts years before the first OOS prediction: score
            # only the sessions the weights engine scores (first book .. last panel date)
            lo, hi = pd.Timestamp(min(books)), pd.Timestamp(panel["date"].max())
            ridx = pd.DatetimeIndex(res.returns.index)
            if ridx.tz is None and lo.tz is not None:
                lo, hi = lo.tz_localize(None), hi.tz_localize(None)
            daily = res.returns[(ridx >= lo) & (ridx <= hi)]
            rec["blocked"] = res.blocked
            rec["stats"] = res.stats
            fills = res.fills
            if len(fills):
                ent = fills[fills["kind"] == "entry"]
                rec["stats"]["short_share_of_entries"] = float((ent["side"] == "short").mean()) if len(ent) else None
        else:
            daily = pc.simulate(v, wide, top_n, True, False, cost_bp)
        idx = pd.DatetimeIndex(daily.index)
        c = cut.tz_localize(idx.tz) if idx.tz is not None else cut
        for seg, series in (("dev", daily[idx < c]), ("holdout", daily[idx >= c]), ("all", daily)):
            m = portfolio_metrics(series)
            if not m:
                continue
            rec[seg] = {k: m[k] for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days")}
            print(f"{name:<12}{seg:<9}{m['sharpe']:>7.2f}{m['ann_return']:>8.1%}{m['max_drawdown']:>8.1%}"
                  f"{ic:>8.4f}{m['n_days']:>8}")
        out["variants"][name] = rec
    suffix = ("" if panel_tag == "ext" else f"_{panel_tag}") + ("" if engine == "weights" else f"_{engine}")
    if engine == "ledger":
        suffix += ("" if apply_caps else "_nocaps") + ("" if brackets else "_nobrackets")
    path = RESULTS / f"production_replay_h{horizon}{suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {path}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    ap.add_argument("--panel", default="surv")
    ap.add_argument("--data", default=None)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--cost-bp", type=float, default=15.0)
    ap.add_argument("--engine", choices=["weights", "ledger"], default="weights")
    ap.add_argument("--no-caps", action="store_true", help="ledger engine: ignore the no-debt / short caps")
    ap.add_argument("--no-brackets", action="store_true", help="ledger engine: no ATR stop / take-profit exits")
    a = ap.parse_args()
    run(a.horizon, a.panel, a.data, top_n=a.top_n, cost_bp=a.cost_bp, engine=a.engine,
        apply_caps=not a.no_caps, brackets=not a.no_brackets)
