"""Pins the pre-registered adoption rulebook v2 (research/evaluation/rulebook.py):
Sidak bar arithmetic, textbook Benjamini-Hochberg, the unchanged track-A gate,
and every hard clause of the track-B (mined) gate."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation import rulebook as rb  # noqa: E402


def _tiers(holdout_t, holdout_ic=None, holdout_n=200, virgin_t=0.5, fresh_t=0.0, fresh_n=30):
    if holdout_ic is None:
        holdout_ic = 0.01 * np.sign(holdout_t) if holdout_t else 0.0
    return {
        "virgin_early": {"ic_mean": 0.003, "t_stat": virgin_t, "n_days": 1200},
        "seen_dev": {"ic_mean": 0.010, "t_stat": 1.5, "n_days": 800},
        "holdout": {"ic_mean": holdout_ic, "t_stat": holdout_t, "n_days": holdout_n},
        "fresh": {"ic_mean": 0.0, "t_stat": fresh_t, "n_days": fresh_n},
    }


# ---- statistics ------------------------------------------------------------
def test_sidak_bar_matches_hand_arithmetic():
    assert abs(rb.sidak_bar(20) - 3.016) < 0.005
    assert abs(rb.sidak_bar(1) - 1.960) < 0.005
    assert rb.sidak_bar(100) > rb.sidak_bar(20)


def test_bh_matches_textbook_example():
    p = [0.01, 0.04, 0.03, 0.20]
    mask = rb.bh_reject(p, q=0.10)
    assert mask.tolist() == [True, True, True, False]
    assert abs(rb.bh_threshold(p, q=0.10) - 0.04) < 1e-12
    assert rb.bh_reject([], q=0.10).size == 0
    assert not rb.bh_reject([0.5, 0.6], q=0.10).any()


def test_onesided_p_respects_declared_direction():
    assert rb.onesided_p(2.0, "positive") < 0.05
    assert rb.onesided_p(2.0, "negative") > 0.95
    assert rb.onesided_p(-2.0, "negative") < 0.05


# ---- families --------------------------------------------------------------
def test_literature_family_excludes_findings(tmp_path):
    led = tmp_path / "ledger.csv"
    pd.DataFrame({
        "date": ["d"] * 4,
        "hypothesis": ["a", "b", "c", "d"],
        "verdict": ["finding: x", "rejected: y", "ADOPTED: z", "no change: w"],
    }).to_csv(led, index=False)
    assert rb.family_size_literature(led) == 3


def test_mined_family_counts_only_full_stage(tmp_path):
    led = tmp_path / "mined.csv"
    pd.DataFrame({
        "stage": ["screen", "full", "full", "screen"],
        "holdout_p_onesided": [np.nan, 0.02, 0.30, np.nan],
    }).to_csv(led, index=False)
    assert rb.family_holdout_pvalues(led) == [0.02, 0.30]
    assert rb.family_holdout_pvalues(tmp_path / "missing.csv") == []


# ---- track A (unchanged) ---------------------------------------------------
def test_track_a_gate_is_the_v1_rule():
    bar = rb.sidak_bar(20)
    assert rb.gate_literature(_tiers(holdout_t=3.2), bar)["verdict"] == "PASS"
    assert rb.gate_literature(_tiers(holdout_t=2.8), bar)["verdict"] == "FAIL"
    # a significantly negative virgin tier vetoes
    assert rb.gate_literature(_tiers(holdout_t=3.2, virgin_t=-2.5), bar)["verdict"] == "FAIL"
    # fresh vetoes only once it has 60 sessions
    assert rb.gate_literature(_tiers(holdout_t=3.2, fresh_t=-2.5, fresh_n=30), bar)["verdict"] == "PASS"
    assert rb.gate_literature(_tiers(holdout_t=3.2, fresh_t=-2.5, fresh_n=60), bar)["verdict"] == "FAIL"


# ---- track B (mined) -------------------------------------------------------
def test_track_b_passes_clean_candidate_in_empty_family():
    out = rb.gate_mined(_tiers(holdout_t=2.5), "positive", blend_dev_gain=0.002,
                        family_pvalues=[])
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["family_n"] == 1


def test_track_b_direction_mismatch_fails():
    out = rb.gate_mined(_tiers(holdout_t=2.5), "negative", blend_dev_gain=0.002)
    assert out["verdict"] == "FAIL"
    assert any("direction" in r for r in out["reasons"])


def test_track_b_requires_dev_blend_gain():
    out = rb.gate_mined(_tiers(holdout_t=2.5), "positive", blend_dev_gain=0.0)
    assert out["verdict"] == "FAIL"
    assert any("blend gain" in r for r in out["reasons"])


def test_track_b_negative_tier_vetoes_in_declared_direction():
    # negative-direction candidate: a POSITIVE virgin t is the harmful sign
    out = rb.gate_mined(_tiers(holdout_t=-2.5, holdout_ic=-0.01, virgin_t=2.5),
                        "negative", blend_dev_gain=0.002)
    assert out["verdict"] == "FAIL"
    assert "virgin_early" in out["negative_tiers"]
    # short fresh tier does not veto
    out = rb.gate_mined(_tiers(holdout_t=2.5, fresh_t=-3.0, fresh_n=20), "positive",
                        blend_dev_gain=0.002)
    assert out["verdict"] == "PASS", out["reasons"]


def test_track_b_holdout_must_be_long_enough():
    out = rb.gate_mined(_tiers(holdout_t=2.5, holdout_n=80), "positive", blend_dev_gain=0.002)
    assert out["verdict"] == "FAIL"


def test_track_b_bh_over_family_tightens_with_many_weak_candidates():
    # p ~ 0.045 passes alone at q=0.10 ...
    alone = rb.gate_mined(_tiers(holdout_t=1.7), "positive", blend_dev_gain=0.002,
                          family_pvalues=[])
    assert alone["verdict"] == "PASS", alone["reasons"]
    # ... but not once 30 prior full-stage candidates with p in (0.3, 0.9) share the family
    crowd = list(np.linspace(0.3, 0.9, 30))
    crowded = rb.gate_mined(_tiers(holdout_t=1.7), "positive", blend_dev_gain=0.002,
                            family_pvalues=crowd)
    assert crowded["verdict"] == "FAIL"
    assert crowded["family_n"] == 31
    # a genuinely strong candidate still clears the crowded family
    strong = rb.gate_mined(_tiers(holdout_t=3.5), "positive", blend_dev_gain=0.002,
                           family_pvalues=crowd)
    assert strong["verdict"] == "PASS", strong["reasons"]
