"""2026-09-14 review, the small self-contained items:

  4b  production predictions fill a missing feature per CELL (as evaluation
      does), never forward-filling one symbol's value into the next symbol
  5   adoption: universe is recorded and a research-universe entry shelves;
      a pool release freezes its members and compiles to the composite in
      production; lookback may be absent
  6   model release: manifest + verify refuse a mixed model set, promote is
      all-or-nothing
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "research"))


# 4b ------------------------------------------------------------------------
class _RecordingModel:
    def __init__(self):
        self.seen = []

    def predict(self, X):
        self.seen.append(np.array(X, dtype=float))
        return np.asarray(X, dtype=float)[:, 0]


def test_production_prediction_fills_per_cell_not_across_symbols(tmp_path, monkeypatch):
    gds = pytest.importorskip("get_daily_signals_multi_factor")
    monkeypatch.setattr(gds, "FACTOR_WEIGHTS", {"fac": 1.0})
    monkeypatch.setattr(gds, "Path", lambda *a, **k: tmp_path / "gds.py")   # factor-score dump goes to tmp, not artifacts/
    dates = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08"])
    rows = []
    for d in dates:
        rows.append({"date": d, "symbol": "AAA", "f1": 5.0, "f2": 1.0, "close": 100.0})
        rows.append({"date": d, "symbol": "BBB", "f1": np.nan, "f2": 2.0, "close": 50.0})   # BBB never has f1
    df = pd.DataFrame(rows)
    model = _RecordingModel()
    ens = {"fac": {"models": {"m": model}, "weights": {"m": 1.0}, "feature_cols": ["f1", "f2"]}}
    out = gds.get_weighted_predictions_multi_factor(df, ens, n_days=5)
    assert model.seen, "the model was never called"
    for X in model.seen:
        assert X.shape == (2, 2)
        assert X[0, 0] == 5.0 and X[1, 0] == 0.0          # BBB's missing f1 is 0, not AAA's 5.0
    assert set(out["symbol"]) == {"AAA", "BBB"}


# 5 -------------------------------------------------------------------------
def _panel(n_dates=30, n_syms=40, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=n_dates)):
        for j in range(n_syms):
            rows.append({"date": d, "symbol": f"S{j:02d}", "close": 10 + rng.normal(), "volume": 1000 + rng.integers(0, 100),
                         "open": 10.0, "high": 11.0, "low": 9.0})
    return pd.DataFrame(rows)


def test_load_adopted_keeps_only_production_horizon_and_universe(tmp_path):
    from factors import mined_factors as mf
    reg = tmp_path / "mined_factors.json"
    reg.write_text(json.dumps({"adopted": [
        {"id": "a", "expression": "rank(close)", "expected_direction": "positive", "horizon": 20},
        {"id": "b", "expression": "rank(close)", "expected_direction": "positive", "horizon": 20, "universe": "sp500"},
        {"id": "c", "expression": "rank(close)", "expected_direction": "positive", "horizon": 20, "universe": "midcap"},
        {"id": "d", "expression": "rank(close)", "expected_direction": "positive", "horizon": 5},
        {"id": "e", "expression": "rank(close)", "expected_direction": "positive", "horizon": 20, "status": "shelf"},
    ]}), encoding="utf-8")
    assert [e["id"] for e in mf.load_adopted(reg)] == ["a", "b"]
    assert [e["id"] for e in mf.load_adopted(reg, production_only=False)] == ["a", "b", "c", "d", "e"]


def test_pool_release_entry_compiles_to_the_frozen_members_composite(tmp_path):
    from factors import mined_factors as mf
    from mining import pool as pl, dsl
    panel = _panel()
    members = [{"hash": "h1", "expression": "rank(close)", "expected_direction": "positive"},
               {"hash": "h2", "expression": "rank(volume)", "expected_direction": "negative"}]
    entry = {"id": "pool_h20_r3", "expression": "pool:h20:2 members", "expected_direction": "positive",
             "horizon": 20, "universe": "sp500", "type": "pool", "members": members}
    got = mf.compile_entry(entry, panel)
    dates = pd.to_datetime(panel["date"]).to_numpy()
    want = pl.composite(pd.DataFrame({
        "h1": pl.signed_rank(dates, dsl.compile_expression("rank(close)", panel).to_numpy(dtype=float), 1.0),
        "h2": pl.signed_rank(dates, dsl.compile_expression("rank(volume)", panel).to_numpy(dtype=float), -1.0)},
        index=panel.index))
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert np.isfinite(got).sum() > 0.9 * len(panel)
    # the production hook adds the column under the entry id
    reg = tmp_path / "mined_factors.json"
    reg.write_text(json.dumps({"adopted": [entry]}), encoding="utf-8")
    out = mf.add_adopted_mined_features(panel, path=reg)
    assert "mined_pool_h20_r3" in out.columns
    np.testing.assert_allclose(out["mined_pool_h20_r3"].to_numpy(), want, equal_nan=True)
    # a member whose source the panel lacks raises instead of zero-filling
    bad = dict(entry, members=members + [{"hash": "h3", "expression": "rank(news_tone)", "expected_direction": "positive"}])
    with pytest.raises(dsl.DSLError):
        mf.compile_entry(bad, panel)


def test_adopt_records_universe_freezes_pool_members_and_shelves_research_universe(tmp_path, monkeypatch):
    from mining import harness, pool as pl
    from evaluation import rulebook as rb
    from factors import mined_factors as mf
    ledger = tmp_path / "mined.csv"
    row = {"date": "2026-09-10", "candidate_id": "pool_h20_midcap_r1", "proposal_hash": "abc", "source": "pool-release:20",
           "expected_direction": "positive", "stage": "full", "verdict": "PASS", "reasons": "",
           "expression": "pool:h20:26 members", "canonical": "pool:h20:26 members", "lookback": np.nan,
           "family_n": 12, "holdout_p_onesided": 0.004, "horizon": 20, "universe": "midcap", "adoption_tier": "structural"}
    rb.append_mined(row, ledger)
    fake_pool = {"members": [{"hash": "h1", "expression": "rank(close)", "expected_direction": "positive", "dev_t": 1.4},
                             {"hash": "h2", "expression": "rank(volume)", "expected_direction": "negative", "dev_t": -1.2}],
                 "releases": []}
    monkeypatch.setattr(pl, "load_pool", lambda h, u="sp500": fake_pool)
    captured = {}
    monkeypatch.setattr(mf, "register_adopted", lambda entry, path=None: captured.update(entry))
    entry = harness.adopt("pool_h20_midcap_r1", "removal: negative fresh tier at the next review", ledger_path=ledger)
    assert captured and entry["universe"] == "midcap" and entry["status"] == "shelf"
    assert entry["lookback"] is None and entry["type"] == "pool" and entry["n_members"] == 2
    assert [m["hash"] for m in entry["members"]] == ["h1", "h2"] and "dev_t" not in entry["members"][0]
    # and the same adoption on the production universe is NOT shelved
    row2 = dict(row, candidate_id="pool_h20_r3", universe="sp500")
    rb.append_mined(row2, ledger)
    entry2 = harness.adopt("pool_h20_r3", "removal: negative fresh tier at the next review", ledger_path=ledger)
    assert entry2["universe"] == "sp500" and "status" not in entry2
    # the shelf entry never reaches production consumers
    reg = tmp_path / "mined_factors.json"
    reg.write_text(json.dumps({"adopted": [entry, entry2]}), encoding="utf-8")
    assert [e["id"] for e in mf.load_adopted(reg)] == ["pool_h20_r3"]


# 6 -------------------------------------------------------------------------
def test_model_release_manifest_verifies_and_refuses_a_mixed_set(tmp_path):
    from factors import model_release as mr
    names = ["momentum", "trend"]
    for n in names:
        (tmp_path / mr.ensemble_file(n)).write_bytes(b"model-" + n.encode())
    (tmp_path / "factor_weights.json").write_text("{}", encoding="utf-8")
    ok, msgs = mr.verify(tmp_path, required=names)
    assert ok and any("UNVERIFIED" in m for m in msgs)                # pre-manifest artifacts: tolerated, flagged
    ok, msgs = mr.verify(tmp_path, required=names + ["high52"])
    assert not ok and "high52" in msgs[-1]
    m = mr.write_manifest(tmp_path, names)
    assert set(m["files"]) == {"ensemble_momentum.pkl", "ensemble_trend.pkl", "factor_weights.json"}
    ok, msgs = mr.verify(tmp_path, required=names)
    assert ok and "verified" in msgs[-1]
    ok, msgs = mr.verify(tmp_path, required=names + ["high52"])
    assert not ok and any("high52" in x for x in msgs)
    (tmp_path / "ensemble_trend.pkl").write_bytes(b"model-trend-from-another-run")
    ok, msgs = mr.verify(tmp_path, required=names)
    assert not ok and any("ensemble_trend.pkl differs" in x for x in msgs)
    (tmp_path / "ensemble_trend.pkl").unlink()
    ok, msgs = mr.verify(tmp_path, required=names)
    assert not ok and any("missing" in x for x in msgs)
    # a release older than the retrain run is not "this run's" release
    (tmp_path / "ensemble_trend.pkl").write_bytes(b"model-trend")
    ok, _ = mr.verify(tmp_path, required=names, not_before=datetime.now() + timedelta(minutes=1))
    assert not ok


def test_model_release_promote_is_all_or_nothing(tmp_path):
    from factors import model_release as mr
    staging = mr.stage_dir(tmp_path)
    staging.mkdir()
    (staging / "ensemble_momentum.pkl").write_bytes(b"new-m")
    (tmp_path / "ensemble_momentum.pkl").write_bytes(b"old-m")
    (tmp_path / "ensemble_trend.pkl").write_bytes(b"old-t")
    with pytest.raises(RuntimeError):
        mr.promote(tmp_path, ["momentum", "trend"])                    # trend never reached staging
    assert (tmp_path / "ensemble_momentum.pkl").read_bytes() == b"old-m"   # nothing moved
    (staging / "ensemble_trend.pkl").write_bytes(b"new-t")
    mr.promote(tmp_path, ["momentum", "trend"])
    assert (tmp_path / "ensemble_momentum.pkl").read_bytes() == b"new-m"
    assert (tmp_path / "ensemble_trend.pkl").read_bytes() == b"new-t" and not list(staging.iterdir())
