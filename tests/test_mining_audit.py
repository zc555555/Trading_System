"""Pins the learning audit (memory.audit_run): a run that re-aims at a
mechanism earlier runs marked dead, or re-submits an indexed expression, is
flagged; a run that reflects, avoids both, and updates beliefs is 'learned'."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import memory  # noqa: E402
from evaluation import rulebook as rb  # noqa: E402


def _ledger(path, rows):
    base = {c: np.nan for c in rb.MINED_COLUMNS}
    pd.DataFrame([{**base, **r} for r in rows], columns=rb.MINED_COLUMNS).to_csv(path, index=False)


def _run(root, name, beliefs=None, proposals=None, reflection=False, age=0):
    d = root / name
    d.mkdir(parents=True)
    if beliefs is not None:
        (d / "beliefs.json").write_text(json.dumps({"run_id": name, "goal": "g", "beliefs": beliefs}), encoding="utf-8")
    if proposals is not None:
        (d / "proposals_1.json").write_text(json.dumps({"proposals": proposals}), encoding="utf-8")
    if reflection:
        (d / "reflection.md").write_text("反省", encoding="utf-8")
    t = time.time() - age
    os.utime(d, (t, t))
    return d


def test_audit_flags_dead_mechanism_and_duplicate(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "cc_r1", beliefs=[{"mechanism": "量价相关", "status": "dead", "evidence": "", "next": ""}], age=100)
    led = tmp_path / "mined.csv"
    _ledger(led, [{"date": "2026-09-01", "candidate_id": "old", "source": "cc-miner:cc_r1", "stage": "screen",
                   "expected_direction": "positive", "canonical": "ts_corr(returns, volume, 20)",
                   "dev_ic": 0.0, "dev_t": 0.1, "coverage": 0.99, "screen_pass": False}])
    r2 = _run(runs, "cc_r2", proposals=[
        {"candidate_id": "again", "expression": "ts_corr(returns,volume,20)", "mechanism_tag": "量价相关"},
        {"candidate_id": "fresh_idea", "expression": "rank(ts_std(returns, 10))", "mechanism_tag": "低波动"}],
        beliefs=[{"mechanism": "低波动", "status": "weak", "evidence": "", "next": ""},
                 {"mechanism": "量价相关", "status": "dead", "evidence": "", "next": ""}])
    a = memory.audit_run(r2, led, runs)
    assert a["n_proposals"] == 2 and a["n_tagged_with_mechanism"] == 2
    assert a["dead_mechanisms_known_before"] == ["量价相关"]
    assert a["proposals_aimed_at_dead_mechanisms"] == ["again"]
    assert a["proposals_duplicating_earlier_expressions"] == ["again"]
    assert a["new_mechanisms_this_run"] == ["低波动"]
    assert a["reflection_written"] is False and a["learned"] is False


def test_audit_passes_a_run_that_learned(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "cc_r1", beliefs=[{"mechanism": "量价相关", "status": "dead", "evidence": "", "next": ""},
                                 {"mechanism": "隔夜反转", "status": "untested", "evidence": "", "next": ""}], age=100)
    led = tmp_path / "mined.csv"
    _ledger(led, [{"date": "2026-09-01", "candidate_id": "old", "source": "cc-miner:cc_r1", "stage": "screen",
                   "expected_direction": "positive", "canonical": "ts_corr(returns, volume, 20)",
                   "dev_ic": 0.0, "dev_t": 0.1, "coverage": 0.99, "screen_pass": False}])
    r2 = _run(runs, "cc_r2", reflection=True, proposals=[
        {"candidate_id": "on_ret", "expression": "neg(ts_sum(open / delay(close, 1) - 1, 10))", "mechanism_tag": "隔夜反转"}],
        beliefs=[{"mechanism": "隔夜反转", "status": "weak", "evidence": "t 1.2", "next": "换窗口"}])
    a = memory.audit_run(r2, led, runs)
    assert a["proposals_aimed_at_dead_mechanisms"] == [] and a["proposals_duplicating_earlier_expressions"] == []
    assert a["beliefs_changed_this_run"] == {"隔夜反转": "untested -> weak"}
    assert a["learned"] is True
