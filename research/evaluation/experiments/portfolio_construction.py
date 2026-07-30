"""Portfolio construction experiment (Layer 2): width x neutrality x cost.

Same signal (the h=5 extended walk-forward panel), same 5-day staggered
execution -- only the CONSTRUCTION changes:

  width      : top-10 (production today) vs top-30 vs top-50 by |pred|
  weighting  : signal-proportional vs signal/vol (inverse-vol tilt)
  neutrality : raw (whatever long/short mix the signal gives) vs
               dollar-neutral (each side scaled to 50% gross)
  cost       : 30bp round trip (old assumption) vs 15bp (calibrated,
               slippage_calibration.csv, paper lower-bound caveat)

IC never changes across the grid; any Sharpe difference is pure
construction. Dev/holdout discipline as usual (holdout = 2025-07+).

Usage:
    python evaluation/experiments/portfolio_construction.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics import portfolio_metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "stocks_with_time_windows.parquet"
HOLDOUT = "2025-07-01"
HOLD_DAYS = 5  # overridden by --horizon
MIN_STOCKS = 3


def load_inputs(horizon: int = 5):
    panel = pd.read_parquet(RESULTS / f"oos_predictions_h{horizon}_ext.parquet")
    prices = pd.read_parquet(DATA, columns=["date", "symbol", "open", "close"])
    prices = prices.sort_values(["symbol", "date"])
    g = prices.groupby("symbol")
    ret = g["close"].pct_change()
    vol20 = (ret.groupby(prices["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(252))
    prices["vol20"] = vol20
    panel = panel.merge(prices[["date", "symbol", "vol20"]],
                        on=["date", "symbol"], how="left")
    sessions = np.sort(prices["date"].unique())
    wide = {
        "sessions": list(sessions),
        "idx": {d: i for i, d in enumerate(sessions)},
        "open": prices.set_index(["date", "symbol"])["open"].to_dict(),
        "close": prices.set_index(["date", "symbol"])["close"].to_dict(),
    }
    return panel, wide


def select(day: pd.DataFrame, top_n: int, inv_vol: bool,
           neutral: bool) -> pd.DataFrame:
    day = day.dropna(subset=["pred"]).assign(conf=lambda d: d["pred"].abs())
    day = day.nlargest(top_n, "conf")
    if len(day) < MIN_STOCKS:
        return day.iloc[0:0]
    raw = day["conf"].copy()
    if inv_vol:
        vol = day["vol20"].clip(lower=0.10).fillna(day["vol20"].median() or 0.3)
        raw = raw / vol
    day = day.assign(weight=raw, direction=np.sign(day["pred"]))
    day = day[day["direction"] != 0]
    cap = max(0.15, 2.0 / top_n)
    if neutral:
        parts = []
        for sign, side_gross in ((1, 0.5), (-1, 0.5)):
            side = day[day["direction"] == sign]
            if len(side):
                w = side["weight"] / side["weight"].sum() * side_gross
                parts.append(side.assign(weight=w.clip(upper=cap)))
        day = pd.concat(parts) if parts else day.iloc[0:0]
    else:
        day = day.assign(
            weight=(day["weight"] / day["weight"].sum()).clip(upper=cap))
    return day


def simulate(panel: pd.DataFrame, wide: dict, top_n: int, inv_vol: bool,
             neutral: bool, cost_bp: float) -> pd.Series:
    sessions, sess_idx = wide["sessions"], wide["idx"]
    open_px, close_px = wide["open"], wide["close"]
    rt = cost_bp / 1e4
    daily = defaultdict(float)
    for date, day in panel.groupby("date", sort=True):
        i = sess_idx.get(date)
        if i is None or i + 1 >= len(sessions):
            continue
        sel = select(day, top_n, inv_vol, neutral)
        for _, r in sel.iterrows():
            sym, w, d = r["symbol"], r["weight"] / HOLD_DAYS, r["direction"]
            entry = open_px.get((sessions[i + 1], sym))
            if entry is None or not np.isfinite(entry) or entry <= 0:
                continue
            prev = entry
            last_h = min(HOLD_DAYS, len(sessions) - 1 - i)
            for h in range(1, last_h + 1):
                px = close_px.get((sessions[i + h], sym))
                if px is None or not np.isfinite(px):
                    break
                mark = d * (px / prev - 1)
                if h == 1:
                    mark -= rt / 2
                if h == last_h:
                    mark -= rt / 2
                daily[sessions[i + h]] += w * mark
                prev = px
    return pd.Series(dict(daily)).sort_index()


def main(horizon: int = 5):
    global HOLD_DAYS
    HOLD_DAYS = horizon
    panel, wide = load_inputs(horizon)
    print(f"panel (h={horizon}, hold={HOLD_DAYS}): {len(panel):,} rows, "
          f"{panel['date'].min().date()} .. {panel['date'].max().date()}")

    grid = []
    for top_n in (10, 30, 50):
        for inv_vol in (False, True):
            for neutral in (False, True):
                for cost_bp in (30.0, 15.0):
                    grid.append((top_n, inv_vol, neutral, cost_bp))

    rows = []
    print(f"\n{'config':<34}{'seg':<9}{'sharpe':>7}{'ann':>8}{'maxDD':>8}")
    print("-" * 68)
    for top_n, inv_vol, neutral, cost_bp in grid:
        daily = simulate(panel, wide, top_n, inv_vol, neutral, cost_bp)
        name = (f"n={top_n:<3} {'ivol' if inv_vol else 'sig '} "
                f"{'ntrl' if neutral else 'raw '} {cost_bp:.0f}bp")
        idx = pd.DatetimeIndex(daily.index)
        cut = pd.Timestamp(HOLDOUT)
        if idx.tz is not None:
            cut = cut.tz_localize(idx.tz)
        for seg, series in (("dev", daily[idx < cut]),
                            ("holdout", daily[idx >= cut])):
            m = portfolio_metrics(series)
            if not m:
                continue
            rows.append({"top_n": top_n, "inv_vol": inv_vol,
                         "neutral": neutral, "cost_bp": cost_bp, "seg": seg,
                         **{k: m[k] for k in
                            ("sharpe", "ann_return", "ann_vol", "max_drawdown",
                             "n_days")}})
            print(f"{name:<34}{seg:<9}{m['sharpe']:>7.2f}"
                  f"{m['ann_return']:>8.1%}{m['max_drawdown']:>8.1%}")

    out = RESULTS / f"portfolio_construction_grid_h{horizon}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5, choices=[5, 20])
    args = ap.parse_args()
    main(horizon=args.horizon)
