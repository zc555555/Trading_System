"""Pins rulebook v3: two adoption paths in one BH family (structural = pooled
unseen tiers; probation = recent holdout+fresh), the common clauses, the
per-row family p-values, and the segment rotation schedule."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation import rulebook as rb  # noqa: E402


def _tiers(virgin=(0.01, 2.0, 1200), holdout=(0.005, 0.5, 220), fresh=(0.01, 1.0, 70), dev=(0.01, 1.5, 800)):
    def d(ic, t, n):
        return {"ic_mean": ic, "t_stat": t, "n_days": n}
    return {"virgin_early": d(*virgin), "seen_dev": d(*dev), "holdout": d(*holdout), "fresh": d(*fresh)}


def test_structural_path_pooled_statistic_and_records():
    pooled = {"ic_mean": 0.008, "t_stat": 3.0, "n_days": 1490}
    recent = {"ic_mean": 0.006, "t_stat": 0.9, "n_days": 290}
    out = rb.gate_mined(_tiers(), "positive", 0.002, [], pooled=pooled, recent=recent)
    assert out["verdict"] == "PASS" and out["adoption_tier"] == "structural", out["reasons"]
    assert out["family_n"] == 2                                     # both of its own p-values are in the family
    assert abs(out["pooled_p_onesided"] - rb.onesided_p(3.0, "positive")) < 1e-12
    assert abs(out["recent_p_onesided"] - rb.onesided_p(0.9, "positive")) < 1e-12
    assert out["holdout_p_onesided"] == rb.onesided_p(0.5, "positive")   # informational


def test_young_effect_takes_the_probation_path():
    # absent before seen_dev, alive since: virgin ~0, holdout and fresh positive
    tiers = _tiers(virgin=(0.0005, 0.1, 1500), holdout=(0.025, 2.4, 250), fresh=(0.03, 1.5, 70))
    pooled = {"ic_mean": 0.004, "t_stat": 1.0, "n_days": 1820}     # diluted by six flat years
    recent = {"ic_mean": 0.026, "t_stat": 2.9, "n_days": 320}
    out = rb.gate_mined(tiers, "positive", 0.002, [], pooled=pooled, recent=recent)
    assert out["verdict"] == "PASS" and out["adoption_tier"] == "probation", out["reasons"]
    # under a pooled-only rule it would have failed: that is the case the user raised
    assert not rb.bh_reject([out["pooled_p_onesided"]], rb.FDR_Q)[0]
    # and a pooled t between the BH cut and the structural floor is still probation, not structural
    mid = rb.gate_mined(tiers, "positive", 0.002, [], pooled={"ic_mean": 0.006, "t_stat": 1.4, "n_days": 1820},
                        recent=recent)
    assert mid["verdict"] == "PASS" and mid["adoption_tier"] == "probation"   # the label is earned by pooled t alone


def test_old_effect_that_died_fails_both_paths():
    tiers = _tiers(virgin=(0.03, 4.0, 1500), holdout=(-0.004, -0.5, 250), fresh=(-0.003, -0.3, 70))
    pooled = {"ic_mean": 0.02, "t_stat": 4.5, "n_days": 1820}
    recent = {"ic_mean": -0.004, "t_stat": -0.6, "n_days": 320}
    out = rb.gate_mined(tiers, "positive", 0.002, [], pooled=pooled, recent=recent)
    assert out["verdict"] == "FAIL"
    joined = " ".join(out["reasons"])
    assert "structural:" in joined and "recent" in joined and "probation:" in joined


def test_h5_loud_day_pattern_under_v3():
    # dev/virgin/fresh negative, holdout flat: passes consistency and (if blend gain) structural
    tiers = _tiers(virgin=(-0.015, -3.3, 1500), holdout=(-0.0025, -0.3, 284), fresh=(-0.0085, -0.6, 63))
    pooled = {"ic_mean": -0.012, "t_stat": -3.4, "n_days": 1847}
    recent = {"ic_mean": -0.004, "t_stat": -0.6, "n_days": 347}
    ok = rb.gate_mined(tiers, "negative", 0.001, [], pooled=pooled, recent=recent)
    assert ok["verdict"] == "PASS" and ok["adoption_tier"] == "structural", ok["reasons"]
    no_gain = rb.gate_mined(tiers, "negative", -0.0001, [], pooled=pooled, recent=recent)
    assert no_gain["verdict"] == "FAIL" and no_gain["reasons"] == ["no blend gain on seen_dev"]


def test_common_clauses_block_both_paths():
    tiers = _tiers(virgin=(-0.02, -2.5, 1500), holdout=(0.02, 2.5, 250), fresh=(0.02, 1.5, 70))
    pooled = {"ic_mean": 0.001, "t_stat": 0.2, "n_days": 1820}
    recent = {"ic_mean": 0.02, "t_stat": 3.0, "n_days": 320}
    out = rb.gate_mined(tiers, "positive", 0.002, [], pooled=pooled, recent=recent)
    assert out["verdict"] == "FAIL" and any("significantly negative" in r for r in out["reasons"])


def test_family_grows_by_two_per_v3_candidate_and_bh_tightens():
    tiers = _tiers(virgin=(0.0005, 0.1, 1500), holdout=(0.02, 2.0, 250), fresh=(0.02, 1.0, 70))
    pooled = {"ic_mean": 0.004, "t_stat": 1.0, "n_days": 1820}
    recent = {"ic_mean": 0.02, "t_stat": 1.75, "n_days": 320}       # p ~ 0.04: passes alone at q=0.10
    alone = rb.gate_mined(tiers, "positive", 0.002, [], pooled=pooled, recent=recent)
    assert alone["verdict"] == "PASS" and alone["adoption_tier"] == "probation"
    crowd = list(np.linspace(0.3, 0.9, 40))                           # 20 earlier v3 candidates, two p each
    crowded = rb.gate_mined(tiers, "positive", 0.002, crowd, pooled=pooled, recent=recent)
    assert crowded["verdict"] == "FAIL" and crowded["family_n"] == 42


def test_legacy_caller_without_series_uses_holdout_and_says_so():
    out = rb.gate_mined(_tiers(holdout=(0.02, 2.6, 220)), "positive", 0.002, [])
    assert out["test_basis"].startswith("holdout") and out["pooled_p_onesided"] is None
    assert out["adoption_tier"] == "probation"                      # only the recent-style test exists


def test_family_pvalues_of_mixes_v2_and_v3_rows():
    rows = pd.DataFrame({"holdout_p_onesided": [0.3, 0.4, 0.5],
                         "pooled_p_onesided": [0.01, np.nan, np.nan],
                         "recent_p_onesided": [0.02, np.nan, 0.06]})
    assert rb.family_pvalues_of(rows) == [0.01, 0.02, 0.4, 0.06]
    assert rb.family_pvalues_of(rows.drop(columns=["pooled_p_onesided", "recent_p_onesided"])) == [0.3, 0.4, 0.5]


def test_segment_rotation_schedule():
    assert rb.SEGMENTS == rb.LEGACY_SEGMENTS and rb.ACTIVE_REVIEW is None
    seg = dict((n, (s, e)) for n, s, e in rb.segments_for("2026-11-15"))
    assert seg["virgin_early"] == (None, "2021-12-30")
    assert seg["seen_dev"] == ("2021-12-30", "2025-11-15")
    assert seg["holdout"] == ("2025-11-15", "2026-11-15")
    assert seg["fresh"] == ("2026-11-15", None)
    assert rb.REVIEW_SCHEDULE[0] == "2026-11-15"
    nxt = dict((n, (s, e)) for n, s, e in rb.segments_for(rb.REVIEW_SCHEDULE[1]))
    assert nxt["holdout"] == ("2026-05-15", "2027-05-15")
