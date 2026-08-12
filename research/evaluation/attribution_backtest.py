"""Full-history return attribution of the production-configuration book.

Re-runs the walk-forward portfolio simulation while collecting the book's
daily net weights, then pushes those weights through the two-layer risk
model (market + sector, risk_model.py) and decomposes every day's return
into:

    market     -- (portfolio beta) x (SPY return)
    sector     -- net sector tilts x sector factor returns
    selection  -- idiosyncratic stock-picking
    costs      -- trading-cost drag (known exactly from the simulation)
    residual   -- execution-timing gap (entry-day open->close marks vs the
                  model's close->close returns); honestly reported, never
                  folded into selection

Three books can be attributed (same code paths as the original
experiments, nothing re-implemented):

  --config prod     (default) the EXACT headline book (all-period Sharpe
                    ~0.8, dev 0.75 / holdout 1.39): the members-only h=20
                    panel (oos_predictions_h20_pitOFF, the ETF-exclusion
                    adoption) through the production construction --
                    top-10 by |pred|, inverse-vol weights, raw long/short
                    mix, 15bp calibrated round-trip costs.
  --config grid     the same construction on the full extended panel
                    (portfolio_construction.py grid population).
  --config baseline the plain simulate_portfolio.py book: |pred|-weights,
                    30bp assumed costs, unextended panel.

Outputs:
    results/attribution_{config}_h{H}.csv
    results/attribution_summary_{config}_h{H}.json
    ../../docs/img/attribution_{config}_h{H}.png

Usage:
    python evaluation/attribution_backtest.py                # prod, h=20
    python evaluation/attribution_backtest.py --config baseline
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evaluation import simulate_portfolio as sim  # noqa: E402
from evaluation.experiments import portfolio_construction as pc  # noqa: E402
from evaluation.risk_model import (  # noqa: E402
    build_risk_model, decompose_portfolio, summarize_attribution,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
HOLDOUT_START = "2025-07-01"

# The adopted production construction (portfolio_construction grid winner).
PROD = {"top_n": 10, "inv_vol": True, "neutral": False, "cost_bp": 15.0}


def _collect_to_frames(collect: dict):
    weights = pd.DataFrame(
        [{"date": dt, "symbol": s, "weight": w}
         for dt, syms in collect["weights"].items() for s, w in syms.items()]
    )
    costs = pd.Series(collect["costs"], name="costs").sort_index()
    return weights, costs


def _load_construction_inputs(panel_path: Path):
    """Mirror portfolio_construction.load_inputs for an arbitrary panel."""
    panel = pd.read_parquet(panel_path)
    prices = pd.read_parquet(pc.DATA, columns=["date", "symbol", "open", "close"])
    prices = prices.sort_values(["symbol", "date"])
    ret = prices.groupby("symbol")["close"].pct_change()
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


def run_construction_book(horizon: int, panel_name: str):
    pc.HOLD_DAYS = horizon
    panel_path = RESULTS_DIR / panel_name
    panel, wide = _load_construction_inputs(panel_path)
    collect: dict = {}
    daily = pc.simulate(panel, wide, collect=collect, **PROD)
    return daily, *_collect_to_frames(collect)


def run_baseline_book(horizon: int):
    panel = sim._load_panel_with_execution_prices(horizon)
    prices = pd.read_parquet(sim.DATA_PATH, columns=["date", "symbol", "open", "close"])
    sessions = np.sort(prices["date"].unique())
    wide = {
        "sessions": list(sessions),
        "open": prices.set_index(["date", "symbol"])["open"].to_dict(),
        "close": prices.set_index(["date", "symbol"])["close"].to_dict(),
    }
    collect: dict = {}
    daily, _trades = sim.simulate_staggered(panel, wide, hold_days=horizon,
                                            collect=collect)
    return daily, *_collect_to_frames(collect)


def segment(attr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cut = pd.Timestamp(HOLDOUT_START)
    idx = pd.DatetimeIndex(attr.index)
    if idx.tz is not None:
        cut = cut.tz_localize(idx.tz)
    return {"dev": attr[idx < cut], "holdout": attr[idx >= cut], "all": attr}


def make_chart(attr: pd.DataFrame, title: str, out_path: Path) -> None:
    comps = ["market", "sector", "selection", "costs", "residual"]
    colors = {"market": "#4878CF", "sector": "#EE854A", "selection": "#6ACC64",
              "costs": "#D65F5F", "residual": "#B47CC7"}
    cum = attr[comps].cumsum()
    total = attr["actual"].cumsum()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    for c in comps:
        ax1.plot(cum.index, cum[c], label=c, color=colors[c], lw=1.4)
    ax1.plot(total.index, total, label="total (actual)", color="black", lw=2.0)
    cut = pd.Timestamp(HOLDOUT_START)
    if cum.index.tz is not None:
        cut = cut.tz_localize(cum.index.tz)
    for ax in (ax1, ax2):
        ax.axvline(cut, color="gray", ls="--", lw=1)
        ax.grid(alpha=0.3)
    ax1.annotate("holdout →", xy=(cut, ax1.get_ylim()[1] * 0.9),
                 fontsize=9, color="gray")
    ax1.set_title(title)
    ax1.set_ylabel("cumulative return")
    ax1.legend(loc="upper left", fontsize=9, ncol=3)

    ax2.plot(attr.index, attr["beta_exposure"], color="#4878CF", lw=1.0)
    ax2.set_ylabel("portfolio beta")
    ax2.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(horizon: int, config: str) -> None:
    print(f"=== attribution: config={config}, h={horizon} ===")
    print("running simulation with weight collection...")
    if config == "prod":
        daily, weights, costs = run_construction_book(
            horizon, f"oos_predictions_h{horizon}_pitOFF.parquet")
        desc = (f"production book (h={horizon}, members universe, "
                f"top-{PROD['top_n']}, inverse-vol, {PROD['cost_bp']:.0f}bp)")
    elif config == "grid":
        daily, weights, costs = run_construction_book(
            horizon, f"oos_predictions_h{horizon}_ext.parquet")
        desc = (f"grid book (h={horizon}, full panel, top-{PROD['top_n']}, "
                f"inverse-vol, {PROD['cost_bp']:.0f}bp)")
    else:
        daily, weights, costs = run_baseline_book(horizon)
        desc = f"baseline book (h={horizon}, |pred|-weights, 30bp)"
    print(f"  sim days: {len(daily)}, weight rows: {len(weights):,}")

    print("building risk model (rolling beta + sector factors)...")
    rm = build_risk_model()
    n_unknown = sum(1 for v in rm.sector_of.values() if v == "Unknown")
    print(f"  {len(rm.sector_of)} symbols, {len(rm.sector_returns.columns)} "
          f"sector buckets ({n_unknown} Unknown)")

    attr = decompose_portfolio(rm, weights, actual_daily=daily, daily_costs=costs)
    stem = f"attribution_{config}_h{horizon}"
    attr.to_csv(RESULTS_DIR / f"{stem}.csv")

    summary = {"generated_at": datetime.now().isoformat(),
               "config": config, "description": desc,
               "horizon": horizon, "holdout_start": HOLDOUT_START,
               "market_proxy": "SPY", "beta_window": 252,
               "segments": {}}
    for name, seg in segment(attr).items():
        s = summarize_attribution(seg)
        summary["segments"][name] = s
        if s.get("n_days"):
            print(f"\n  [{name}] {s['n_days']} days | "
                  f"ann total {s['ann_return_total']:+.1%}")
            for c in ("market", "sector", "selection", "costs", "residual"):
                key = f"ann_return_{c}"
                if key in s:
                    print(f"    {c:<10} {s[key]:+7.1%}")
            print(f"    avg beta exposure {s['avg_beta_exposure']:+.2f}  "
                  f"gross {s['avg_gross_weight']:.0%}  net {s['avg_net_weight']:+.0%}")
            if "var_share_systematic" in s:
                print(f"    variance share systematic (mkt+sector): "
                      f"{s['var_share_systematic']:.0%}")

    with open(RESULTS_DIR / f"{stem}.json".replace(stem, f"attribution_summary_{config}_h{horizon}"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    chart = IMG_DIR / f"{stem}.png"
    make_chart(attr, f"Cumulative return attribution -- {desc}", chart)
    print(f"\nsaved: {stem}.csv / attribution_summary_{config}_h{horizon}.json")
    print(f"chart: {chart}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=20, choices=[1, 5, 20])
    ap.add_argument("--config", choices=["prod", "grid", "baseline"],
                    default="prod")
    args = ap.parse_args()
    main(args.horizon, args.config)
