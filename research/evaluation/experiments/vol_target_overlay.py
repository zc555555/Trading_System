"""Volatility-target overlay experiment (pro-upgrade item 2).

Mechanism under test: each session, BEFORE the day's new tranche is
deployed, forecast the annualized vol of yesterday's (scaled) book with
the two-layer risk model's factor covariance (VolForecaster, data through
t-1 only) and scale TODAY'S NEW TRANCHE by

    m_t = clip(TARGET_ANN_VOL / forecast, 0, 1)      # CAP 1.0, HARD

Scale-down-only is the no-debt principle applied to overlays: exposure
may shrink below 1x, never grow above it.  Only the new tranche is
scaled (existing tranches ride to schedule), because that is the one
knob the staggered architecture can turn without paying turnover -- the
overlay is therefore COSTLESS by construction and the simulation charges
the same per-tranche costs, scaled.

PRE-REGISTERED CONFIG (declared before results were seen; no sweeping):
    TARGET_ANN_VOL = 0.12, forecaster window 120d (min 60), cap 1.0.

PRE-REGISTERED ADOPTION RULE (a risk overlay, not an alpha -- the bar is
risk-shape improvement without return-quality damage), all on dev:
    (1) realized ann vol moves toward target by >= 1 vol point;
    (2) max drawdown improves (shallower);
    (3) Sharpe does not degrade by more than 0.05.
If all three pass on dev, holdout must CONFIRM direction: vol lower than
baseline and Sharpe not degraded by more than 0.10 (wider band, 215 days
of noise).  Anything else -> reject and record in the ledger.

Usage:
    python evaluation/experiments/vol_target_overlay.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from evaluation.attribution_backtest import (  # noqa: E402
    PROD, _load_construction_inputs,
)
from evaluation.experiments import portfolio_construction as pc  # noqa: E402
from evaluation.metrics import portfolio_metrics  # noqa: E402
from evaluation.risk_model import VolForecaster, build_risk_model  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
HOLDOUT = "2025-07-01"
HORIZON = 20

TARGET_ANN_VOL = 0.12
FORECAST_WINDOW = 120
CAP = 1.0                     # scale-down only; pinned by the no-debt tests


def run_overlay():
    pc.HOLD_DAYS = HORIZON
    panel, wide = _load_construction_inputs(
        RESULTS / f"oos_predictions_h{HORIZON}_pitOFF.parquet")
    collect: dict = {}
    baseline_daily = pc.simulate(panel, wide, collect=collect, **PROD)

    # tranche detail: entry session -> {mark session -> pnl / weights}
    pnl_by_entry: dict = {}
    w_by_entry: dict = {}
    for (entry, mark), v in collect["tranche_pnl"].items():
        pnl_by_entry.setdefault(entry, {})[mark] = v
    for (entry, mark), syms in collect["tranche_weights"].items():
        w_by_entry.setdefault(entry, {})[mark] = syms

    print("building risk model + vol forecaster...")
    rm = build_risk_model()
    vf = VolForecaster(rm, window=FORECAST_WINDOW)

    sessions = [d for d in vf.dates if d in set(baseline_daily.index)
                or d in pnl_by_entry]
    sessions = sorted(set(sessions))

    scale: dict = {}                 # entry session -> multiplier
    scaled_daily: dict = {}
    multipliers: dict = {}
    forecasts: dict = {}

    for t in sessions:
        # 1) yesterday's book, scaled: all tranches whose life covers t
        book: dict = {}
        for entry, marks in w_by_entry.items():
            if t in marks:
                s = scale.get(entry, 1.0)
                for sym, w in marks[t].items():
                    book[sym] = book.get(sym, 0.0) + s * w
        # 2) forecast its vol with data through t-1; set today's multiplier
        fv = vf.forecast_ann_vol(t, book) if book else None
        m = CAP if fv is None or fv <= 0 else min(CAP, TARGET_ANN_VOL / fv)
        if t in pnl_by_entry:        # a tranche enters at t -> scale it
            scale[t] = m
        multipliers[t] = m
        if fv is not None:
            forecasts[t] = fv
        # 3) realized return of the scaled book at t
        r = 0.0
        for entry, marks in pnl_by_entry.items():
            if t in marks:
                r += scale.get(entry, 1.0) * marks[t]
        if t in set(baseline_daily.index):
            scaled_daily[t] = r

    return (baseline_daily,
            pd.Series(scaled_daily).sort_index(),
            pd.Series(multipliers).sort_index(),
            pd.Series(forecasts).sort_index())


def seg_split(series: pd.Series):
    idx = pd.DatetimeIndex(series.index)
    cut = pd.Timestamp(HOLDOUT)
    if idx.tz is not None:
        cut = cut.tz_localize(idx.tz)
    return {"dev": series[idx < cut], "holdout": series[idx >= cut]}


def main():
    print(f"=== vol-target overlay: target {TARGET_ANN_VOL:.0%}, "
          f"window {FORECAST_WINDOW}d, cap {CAP} ===")
    baseline, overlay, mult, fcast = run_overlay()

    report = {"generated_at": datetime.now().isoformat(),
              "config": {"target_ann_vol": TARGET_ANN_VOL,
                         "forecast_window": FORECAST_WINDOW, "cap": CAP,
                         "book": "prod pitOFF h20", **PROD},
              "segments": {}}

    print(f"\n{'seg':<9}{'variant':<10}{'sharpe':>7}{'ann':>8}{'vol':>7}"
          f"{'maxDD':>8}{'avg_m':>7}")
    print("-" * 56)
    checks = {}
    for seg in ("dev", "holdout"):
        b = portfolio_metrics(seg_split(baseline)[seg])
        o = portfolio_metrics(seg_split(overlay)[seg])
        am = float(seg_split(mult)[seg].mean())
        report["segments"][seg] = {"baseline": b, "overlay": o,
                                   "avg_multiplier": am}
        for name, m in (("baseline", b), ("overlay", o)):
            print(f"{seg:<9}{name:<10}{m['sharpe']:>7.2f}{m['ann_return']:>8.1%}"
                  f"{m['ann_vol']:>7.1%}{m['max_drawdown']:>8.1%}"
                  f"{am if name == 'overlay' else 1.0:>7.2f}")
        checks[seg] = {"b": b, "o": o}

    d_b, d_o = checks["dev"]["b"], checks["dev"]["o"]
    h_b, h_o = checks["holdout"]["b"], checks["holdout"]["o"]
    c1 = (d_b["ann_vol"] - d_o["ann_vol"]) >= 0.01 and \
        abs(d_o["ann_vol"] - TARGET_ANN_VOL) < abs(d_b["ann_vol"] - TARGET_ANN_VOL)
    c2 = d_o["max_drawdown"] > d_b["max_drawdown"]      # less negative
    c3 = d_o["sharpe"] >= d_b["sharpe"] - 0.05
    dev_pass = c1 and c2 and c3
    c4 = h_o["ann_vol"] < h_b["ann_vol"] and h_o["sharpe"] >= h_b["sharpe"] - 0.10
    verdict = "ADOPT" if (dev_pass and c4) else "REJECT"

    print(f"\npre-registered checks (dev): "
          f"vol-toward-target {'PASS' if c1 else 'FAIL'} | "
          f"maxDD-improves {'PASS' if c2 else 'FAIL'} | "
          f"sharpe-within-0.05 {'PASS' if c3 else 'FAIL'}")
    print(f"holdout confirmation: {'PASS' if c4 else 'FAIL'}")
    print(f"\nVERDICT: {verdict}")

    report["checks"] = {"dev_vol_toward_target": bool(c1),
                        "dev_maxdd_improves": bool(c2),
                        "dev_sharpe_within_0p05": bool(c3),
                        "holdout_confirms": bool(c4),
                        "verdict": verdict}
    report["forecast_stats"] = {
        "mean": float(fcast.mean()), "p10": float(fcast.quantile(0.1)),
        "p90": float(fcast.quantile(0.9)), "n": int(len(fcast))}

    out = RESULTS / "vol_target_overlay_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    pd.DataFrame({"multiplier": mult, "forecast_vol": fcast}).to_csv(
        RESULTS / "vol_target_overlay_series.csv")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
