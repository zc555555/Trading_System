"""Pins the per-horizon track-B machinery: configure() switches label / lags /
cache, proposals must match the harness horizon, duplicates and families are
per horizon, beliefs merge per (mechanism, horizon), and shelf adoptions
never reach production consumers."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import harness, memory  # noqa: E402
from mining.proposal import Proposal  # noqa: E402
from evaluation import rulebook as rb  # noqa: E402
from factors import mined_factors as mf  # noqa: E402


@pytest.fixture
def h5():
    harness.configure(5)
    yield
    harness.configure(20)


def test_configure_switches_label_lags_and_cache(h5):
    assert harness.HORIZON == 5 and harness.NW_LAGS == 4 and harness.LABEL == "future_return_5d"
    assert harness.screen_cache_path().name.endswith("_h5.parquet")
    harness.configure(20)
    assert harness.LABEL == "future_return_20d" and harness.screen_cache_path().name == "_mining_screen_panel_v2.parquet"
    with pytest.raises(ValueError):
        harness.configure(7)


def test_add_label_uses_the_configured_horizon(h5):
    dates = pd.bdate_range("2024-01-01", periods=30, tz="America/New_York")
    df = pd.DataFrame({"date": dates, "symbol": "A", "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.arange(1, 31, dtype=float), "volume": 1.0})
    out = harness.add_label(df)
    assert "future_return_5d" in out.columns
    assert abs(out.loc[0, "future_return_5d"] - np.log(6 / 1)) < 1e-12


def test_proposal_horizon_must_match_the_harness(h5):
    p20 = Proposal(candidate_id="p20", expression="rank(close)", expected_direction="positive",
                   hypothesis="A proposal written for the production horizon.",
                   mechanism="No mechanism; this exists to test the horizon check.",
                   refutation_conditions=["x"], horizon=20)
    p7 = Proposal(candidate_id="p7", expression="rank(close)", expected_direction="positive",
                  hypothesis="A proposal at an unsupported horizon.",
                  mechanism="No mechanism; this exists to test the horizon check.",
                  refutation_conditions=["x"], horizon=7)
    with pytest.raises(ValueError, match="5 and 20"):
        p7.validate()
    panel = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5, tz="America/New_York"), "symbol": "A",
                          "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                          "future_return_5d": 0.0, "pit": True})
    row = harness.screen_one(p20, panel, Path("nonexistent.csv"), record=False)
    assert "horizon 20 != harness horizon 5" in row["error"]


def test_family_and_duplicates_are_per_horizon(tmp_path):
    led = tmp_path / "mined.csv"
    base = {c: np.nan for c in rb.MINED_COLUMNS}
    rb.append_mined({**base, "candidate_id": "a", "stage": "full", "holdout_p_onesided": 0.01, "horizon": 20}, led)
    rb.append_mined({**base, "candidate_id": "b", "stage": "full", "holdout_p_onesided": 0.02, "horizon": 5}, led)
    rb.append_mined({**base, "candidate_id": "c", "stage": "full", "holdout_p_onesided": 0.03}, led)   # legacy row = 20
    assert sorted(rb.family_holdout_pvalues(led)) == [0.01, 0.02, 0.03]
    assert rb.family_holdout_pvalues(led, horizon=20) == [0.01, 0.03]
    assert rb.family_holdout_pvalues(led, horizon=5) == [0.02]
    rb.append_mined({**base, "candidate_id": "x20", "stage": "screen", "canonical": "rank(close)", "dev_t": 1.0,
                     "screen_pass": False, "horizon": 20, "date": "2026-09-03"}, led)
    assert memory.find_duplicate(led, "rank(close)", horizon=20)["candidate_id"] == "x20"
    assert memory.find_duplicate(led, "rank(close)", horizon=5) is None


def test_beliefs_merge_per_mechanism_and_horizon(tmp_path):
    r1 = tmp_path / "cc_r1"; r1.mkdir()
    (r1 / "beliefs.json").write_text(json.dumps({"run_id": "cc_r1", "horizon": 20, "beliefs": [
        {"mechanism": "量价", "status": "dead", "evidence": "e", "next": "n"}]}), encoding="utf-8")
    r2 = tmp_path / "cc_r2"; r2.mkdir()
    (r2 / "beliefs.json").write_text(json.dumps({"run_id": "cc_r2", "horizon": 5, "beliefs": [
        {"mechanism": "量价", "status": "promising", "evidence": "e5", "next": "n5"}]}), encoding="utf-8")
    beliefs, runs = memory.load_beliefs(tmp_path)
    assert beliefs[("量价", 20)]["status"] == "dead" and beliefs[("量价", 5)]["status"] == "promising"
    doc = memory.build_memory(tmp_path / "none.csv", tmp_path, horizon=5)
    assert "| 量价 | 20 | dead |" in doc and "| 量价 | 5 | promising |" in doc and "本轮持有期 5 日" in doc


def test_shelf_adoptions_never_reach_production(tmp_path):
    reg = tmp_path / "mined_factors.json"
    mf.register_adopted({"id": "prod_one", "expression": "rank(close)", "horizon": 20}, reg)
    mf.register_adopted({"id": "shelf_one", "expression": "rank(open)", "horizon": 5, "status": "shelf"}, reg)
    assert [e["id"] for e in mf.load_adopted(reg)] == ["prod_one"]
    assert [e["id"] for e in mf.load_adopted(reg, production_only=False)] == ["prod_one", "shelf_one"]
    assert mf.adopted_feature_columns(reg) == ["mined_prod_one"]
    assert list(mf.mined_factor_groups(reg)) == ["prod_one"]
