"""Pins the factor report card's core checks (research/evaluation/factor_card.py):
decile monotonicity on a clean signal, residual IC that vanishes for a
relabelled copy of another factor, and turnover of a persistent score."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation.factor_card import (  # noqa: E402
    decile_table, per_date_zscore, residual_ic, turnover,
)

RNG = np.random.default_rng(3)


def _panel(n_dates=60, n_syms=120):
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    rows = []
    for d in dates:
        x = RNG.normal(size=n_syms)                # true signal
        noise = RNG.normal(size=n_syms)
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"S{i}" for i in range(n_syms)],
            "factor_true": x,
            "factor_copy": 2.0 * x + 0.05 * noise,   # relabelled copy
            "factor_indep": noise,                   # unrelated
            "future_return_20d": 0.01 * x + 0.02 * RNG.normal(size=n_syms),
        }))
    return pd.concat(rows, ignore_index=True)


def test_deciles_are_monotonic_for_a_clean_signal():
    df = _panel()
    d = decile_table(df, "factor_true", "future_return_20d")
    assert d["n_dates"] == 60
    assert d["monotonicity"] > 0.9
    assert d["spread_mean"] > 0 and d["spread_t"] > 3


def test_residual_ic_vanishes_for_a_copy_but_survives_for_independent_info():
    df = _panel()
    df[["factor_true", "factor_copy", "factor_indep"]] = per_date_zscore(
        df, ["factor_true", "factor_copy", "factor_indep"])
    raw_copy = df.groupby("date").apply(
        lambda g: g["factor_copy"].corr(g["future_return_20d"], method="spearman"),
        include_groups=False).mean()
    resid_copy = residual_ic(df, "factor_copy", ["factor_true"], "future_return_20d").mean()
    assert raw_copy > 0.3
    assert abs(resid_copy) < 0.1, "a relabelled copy must carry ~no residual information"
    resid_true = residual_ic(df, "factor_true", ["factor_indep"], "future_return_20d").mean()
    assert resid_true > 0.3, "orthogonalising against unrelated noise must not remove the signal"


def test_turnover_low_for_persistent_score_high_for_noise():
    dates = pd.bdate_range("2023-01-02", periods=50)
    syms = [f"S{i}" for i in range(100)]
    persistent = RNG.normal(size=100)
    rows = []
    for d in dates:
        rows.append(pd.DataFrame({"date": d, "symbol": syms,
                                  "factor_persistent": persistent + 0.01 * RNG.normal(size=100),
                                  "factor_noise": RNG.normal(size=100)}))
    df = pd.concat(rows, ignore_index=True)
    lo = turnover(df, "factor_persistent", hold=20)["turnover_per_hold"]
    hi = turnover(df, "factor_noise", hold=20)["turnover_per_hold"]
    assert lo < 0.2 and hi > 0.7
