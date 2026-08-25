"""Pins the pre-registered alert rules of the IC decay monitor
(research/evaluation/ic_monitor.py): WARN needs a sustained drop below
the dev band, ALERT needs a sustained negative rolling IC AND a live
production weight, and per-date IC is a cross-sectional rank correlation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation.ic_monitor import (  # noqa: E402
    daily_factor_ic, dev_baseline, evaluate_alerts,
)


def _rolling(values_by_factor: dict, n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({f: np.asarray(v, dtype=float)
                         for f, v in values_by_factor.items()}, index=idx)


def test_daily_ic_is_cross_sectional_rank_correlation():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=5):
        x = rng.normal(size=60)
        rows.append(pd.DataFrame({"date": d, "symbol": range(60),
                                  "factor_a": x, "factor_b": -x,
                                  "future_return_20d": x + rng.normal(scale=0.1, size=60)}))
    panel = pd.concat(rows)
    ics = daily_factor_ic(panel, ["factor_a", "factor_b"], "future_return_20d")
    assert len(ics) == 5
    assert (ics["factor_a"] > 0.9).all() and (ics["factor_b"] < -0.9).all()
    # matches scipy-free reference: spearman of one date
    d0 = panel[panel["date"] == panel["date"].min()]
    ref = d0["factor_a"].corr(d0["future_return_20d"], method="spearman")
    assert np.isclose(ics["factor_a"].iloc[0], ref)


def test_warn_requires_sustained_drop_below_band():
    n = 400
    base = np.full(n, 0.02)
    warn_series = base.copy()
    warn_series[-25:] = -0.05          # 25 sessions below the band -> WARN
    blip_series = base.copy()
    blip_series[-5:] = -0.05           # 5 sessions -> still OK
    roll = _rolling({"factor_x": warn_series, "factor_y": blip_series}, n)
    baseline = pd.DataFrame({"mean": [0.02, 0.02], "std": [0.01, 0.01]},
                            index=["factor_x", "factor_y"])
    rep = evaluate_alerts(roll, baseline, {"x": 0.3, "y": 0.3})
    assert rep["factor_x"]["status"] == "WARN"
    assert rep["factor_y"]["status"] == "OK"


def test_alert_needs_negative_run_and_live_weight():
    n = 400
    neg = np.full(n, 0.02)
    neg[-70:] = -0.01                  # 70 sessions below zero
    roll = _rolling({"factor_live": neg, "factor_dead": neg}, n)
    baseline = pd.DataFrame({"mean": [0.02, 0.02], "std": [0.01, 0.01]},
                            index=["factor_live", "factor_dead"])
    rep = evaluate_alerts(roll, baseline, {"live": 0.25, "dead": 0.0})
    assert rep["factor_live"]["status"] == "ALERT"
    assert rep["factor_live"]["sessions_below_zero"] == 70
    # a zero-weight factor is already out of the book: WARN at most
    assert rep["factor_dead"]["status"] == "WARN"


def test_dev_baseline_uses_only_dev_window():
    idx = pd.bdate_range("2021-01-01", "2026-06-30")
    s = pd.Series(0.0, index=idx)
    s[(idx >= "2022-01-01") & (idx < "2025-07-01")] = 0.05
    b = dev_baseline(pd.DataFrame({"factor_z": s}))
    assert np.isclose(b.loc["factor_z", "mean"], 0.05)
    assert np.isclose(b.loc["factor_z", "std"], 0.0)
