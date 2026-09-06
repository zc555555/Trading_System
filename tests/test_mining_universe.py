"""Pins the research-universe switch: configure() selects membership, panel,
baseline tag and cache suffix; rows record the universe; families and
duplicates are per (horizon, universe); pool files carry the suffix; the
guard accepts --universe; membership_mask takes any membership table."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation import rulebook as rb  # noqa: E402
from evaluation.experiments.pit_universe_test import membership_mask  # noqa: E402
from mining import harness, pool as pl  # noqa: E402
from mining.proposal import Proposal  # noqa: E402


def test_configure_switches_paths_and_restores():
    harness.configure(20, "midcap")
    assert harness.UNIVERSE == "midcap" and harness.MEMBERSHIP.name == "midcap_membership.parquet"
    assert harness.SURV_PANEL.name == "stocks_with_time_windows_midcap.parquet" and harness.BASELINE_TAG == "surv_midcap"
    assert harness.screen_cache_path().name == "_mining_screen_panel_v2_midcap.parquet"
    assert harness.aux_sources()["news_source"].name == "nonexistent"
    harness.configure(5, "sp500")
    assert harness.UNIVERSE == "sp500" and harness.MEMBERSHIP.name == "sp500_membership.parquet"
    assert harness.screen_cache_path().name == "_mining_screen_panel_v2_h5.parquet" and harness.aux_sources() == {}
    harness.configure(20, "sp500")
    try:
        harness.configure(20, "nasdaq")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown universe must be refused")


def test_rows_and_families_are_per_universe(tmp_path):
    p = Proposal(candidate_id="u_test", expression="rank(close)", expected_direction="positive",
                 hypothesis="a sentence long enough to pass validation here", mechanism="another sentence long enough to pass",
                 refutation_conditions=["x"], horizon=20)
    harness.configure(20, "midcap")
    row = harness.base_row(p, "screen")
    assert row["universe"] == "midcap" and row["horizon"] == 20
    harness.configure(20, "sp500")
    led = pd.DataFrame([{"candidate_id": "a", "stage": "full", "horizon": 20, "universe": "midcap", "pooled_p_onesided": 0.01, "recent_p_onesided": 0.02},
                        {"candidate_id": "b", "stage": "full", "horizon": 20, "universe": "sp500", "pooled_p_onesided": 0.3, "recent_p_onesided": 0.4},
                        {"candidate_id": "c", "stage": "full", "horizon": 20, "pooled_p_onesided": 0.5, "recent_p_onesided": 0.6}])
    path = tmp_path / "led.csv"; led.to_csv(path, index=False)
    assert sorted(harness.prior_family("zzz", path)) == [0.3, 0.4, 0.5, 0.6]        # sp500 rows + legacy rows only
    harness.configure(20, "midcap")
    assert sorted(harness.prior_family("zzz", path)) == [0.01, 0.02]
    harness.configure(20, "sp500")
    assert list(rb.universe_of(led)) == ["midcap", "sp500", "sp500"]


def test_pool_paths_carry_the_universe(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "POOL_DIR", tmp_path)
    assert pl.pool_path(20).name == "pool_h20.json" and pl.pool_path(20, "midcap").name == "pool_h20_midcap.json"
    assert pl.features_path(5, "midcap").name == "pool_features_h5_midcap.parquet"
    pool = pl.load_pool(20, "midcap")
    assert pool["universe"] == "midcap" and pool["members"] == []
    pl.save_pool(pool)
    assert (tmp_path / "pool_h20_midcap.json").exists()


def test_membership_mask_accepts_another_table(tmp_path):
    mem = pd.DataFrame({"symbol": ["AAA"], "start": [pd.Timestamp("2024-01-03")], "end": [pd.Timestamp("2024-01-05")]})
    path = tmp_path / "mem.parquet"; mem.to_parquet(path, index=False)
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=6, tz="America/New_York"), "symbol": "AAA"})
    m = membership_mask(df, membership_path=path)
    assert m.tolist() == [False, False, True, True, False, False]


def test_guard_accepts_universe_flag():
    sys.path.append(str(Path(__file__).resolve().parent.parent / ".claude" / "hooks"))
    import miner_guard as g
    py = "research/venv/Scripts/python.exe"
    assert g.BASH_OPS.match(f"{py} research/mining/harness.py --horizon 5 --universe midcap ops")
    assert g.BASH_OPS.match(f"{py} research/mining/harness.py --universe midcap memory --part 2")
    assert not g.BASH_OPS.match(f"{py} research/mining/harness.py --universe nasdaq ops")


def test_proposal_carries_and_validates_universe():
    p = Proposal(candidate_id="u_mid", expression="rank(close)", expected_direction="positive",
                 hypothesis="a sentence long enough to pass validation here", mechanism="another sentence long enough to pass",
                 refutation_conditions=["x"], horizon=20, universe="midcap")
    assert p.validate()["canonical"] == "rank(close)"
    bad = Proposal(candidate_id="u_bad", expression="rank(close)", expected_direction="positive",
                   hypothesis="a sentence long enough to pass validation here", mechanism="another sentence long enough to pass",
                   refutation_conditions=["x"], horizon=20, universe="nasdaq")
    try:
        bad.validate()
    except Exception as e:
        assert "universe" in str(e)
    else:
        raise AssertionError("unknown universe must fail validation")
