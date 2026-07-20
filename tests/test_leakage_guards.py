"""Leakage regression guards.

These tests exist because of real bugs found in the 2026-07 audit:
- dpo_20 used shift(-11) and injected 11 future days into a live feature
- clean_dataset backward-filled warm-up rows with future values

They are cheap static/behavioral checks that fail loudly if either
pattern is ever reintroduced.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent / "research"

# Feature-computation modules that must never reference future rows.
FEATURE_MODULES = [
    RESEARCH / "features" / "operators.py",
    RESEARCH / "features" / "alphas_101_subset.py",
    RESEARCH / "features" / "cross_sectional.py",
    RESEARCH / "features" / "add_time_windows.py",
]


def test_no_negative_shift_in_feature_modules():
    for path in FEATURE_MODULES:
        source = path.read_text(encoding="utf-8")
        offending = [
            line.strip()
            for line in source.splitlines()
            if "shift(-" in line and not line.strip().startswith("#")
        ]
        assert not offending, (
            f"{path.name} contains negative shift(s) -- a feature reading "
            f"future rows: {offending}"
        )


def test_negative_shift_in_build_dataset_only_for_labels():
    source = (RESEARCH / "features" / "build_dataset.py").read_text(encoding="utf-8")
    offending = [
        line.strip()
        for line in source.splitlines()
        if "shift(-" in line and not line.strip().startswith("#")
        and "horizon" not in line
    ]
    assert not offending, (
        "build_dataset.py uses shift(-...) outside label construction "
        f"(labels shift by -horizon): {offending}"
    )


def test_no_bfill_in_cleaning_paths():
    for rel in ["features/build_dataset.py", "prepare_prediction_data.py"]:
        source = (RESEARCH / rel).read_text(encoding="utf-8")
        offending = [
            line.strip()
            for line in source.splitlines()
            if ".bfill(" in line and not line.strip().startswith("#")
        ]
        assert not offending, (
            f"{rel} backward-fills again -- bfill copies future values into "
            f"the past: {offending}"
        )


def test_dpo_uses_only_past_data():
    from features import operators

    rng = np.random.default_rng(0)
    n = 120
    base = pd.DataFrame({
        "symbol": ["AAA"] * n,
        "close": 100 + np.cumsum(rng.normal(0, 1, n)),
    })
    dpo_a = np.asarray(operators.dpo(base, "close", 20)).ravel()

    tail = 15
    mut = base.copy()
    mut.loc[mut.index[-tail:], "close"] += 50.0
    dpo_b = np.asarray(operators.dpo(mut, "close", 20)).ravel()

    head_a, head_b = dpo_a[: n - tail], dpo_b[: n - tail]
    valid = ~np.isnan(head_a)
    assert np.array_equal(head_a[valid], head_b[valid]), (
        "dpo values changed when only FUTURE prices were perturbed"
    )

    # Textbook formula: dpo[t] = close[t - (w//2 + 1)] - SMA_w[t]
    t = 80
    expect = base["close"].iloc[t - 11] - base["close"].iloc[t - 19: t + 1].mean()
    assert np.isclose(dpo_a[t], expect)
