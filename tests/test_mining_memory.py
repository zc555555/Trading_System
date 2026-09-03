"""Pins the cross-run research memory (research/mining/memory.py): beliefs
merge with latest-wins, the expression index carries dev fields only, hidden
tier numbers never reach the document, duplicates are detected, and the
harness screen short-circuits a repeated canonical expression."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import memory, harness, dsl  # noqa: E402
from mining.proposal import Proposal  # noqa: E402
from evaluation import rulebook as rb  # noqa: E402

HIDDEN_NUMBERS = ("0.0777", "0.0888", "0.0999", "3.33", "0.00123")


def _ledger(path):
    rows = []
    base = {c: np.nan for c in rb.MINED_COLUMNS}
    rows.append({**base, "date": "2026-09-01", "candidate_id": "a1", "source": "cc-miner:cc_run1",
                 "expected_direction": "positive", "stage": "screen", "dev_ic": 0.011, "dev_t": 0.95,
                 "coverage": 0.99, "screen_pass": False, "canonical": "ts_sum(returns, 5)",
                 "expression": "ts_sum(returns,5)", "verdict": "screen_fail"})
    rows.append({**base, "date": "2026-09-02", "candidate_id": "b1", "source": "cc-miner:cc_run2",
                 "expected_direction": "negative", "stage": "screen", "dev_ic": -0.03, "dev_t": -2.4,
                 "coverage": 0.98, "screen_pass": True, "canonical": "rank(ts_corr(close, volume, 20))",
                 "expression": "rank(ts_corr(close, volume, 20))", "verdict": "screen_pass"})
    rows.append({**base, "date": "2026-09-02", "candidate_id": "b1", "source": "cc-miner:cc_run2",
                 "expected_direction": "negative", "stage": "full", "dev_ic": -0.03, "dev_t": -2.4,
                 "blend_dev_gain": 0.00123, "virgin_ic": 0.0777, "holdout_ic": 0.0888, "holdout_t": 3.33,
                 "fresh_ic": 0.0999, "holdout_p_onesided": 0.0005, "family_n": 1, "bh_threshold": 0.0005,
                 "verdict": "PASS", "reasons": "", "canonical": "rank(ts_corr(close, volume, 20))",
                 "expression": "rank(ts_corr(close, volume, 20))"})
    pd.DataFrame(rows, columns=rb.MINED_COLUMNS).to_csv(path, index=False)


def _runs(root):
    r1 = root / "cc_run1"; r1.mkdir(parents=True)
    (r1 / "beliefs.json").write_text(json.dumps({"run_id": "cc_run1", "goal": "g1", "beliefs": [
        {"mechanism": "量价相关", "status": "weak", "evidence": "t 0.9", "next": "换窗口"},
        {"mechanism": "隔夜反转", "status": "untested", "evidence": "", "next": "试 20 日"}]}), encoding="utf-8")
    (r1 / "notes.md").write_text("# run1 笔记\n量价相关很弱。", encoding="utf-8")
    r2 = root / "cc_run2"; r2.mkdir()
    (r2 / "beliefs.json").write_text(json.dumps({"run_id": "cc_run2", "goal": "g2", "beliefs": [
        {"mechanism": "量价相关", "status": "dead", "evidence": "20/60 均≈0", "next": "放弃"}]}), encoding="utf-8")
    (r2 / "notes.md").write_text("# run2 笔记\n" + "很长的笔记。" * 2000, encoding="utf-8")
    import os, time
    os.utime(r1, (time.time() - 100, time.time() - 100))   # run1 older than run2
    return root


def test_memory_merges_beliefs_latest_wins_and_indexes_expressions(tmp_path):
    led = tmp_path / "mined.csv"; _ledger(led)
    runs = _runs(tmp_path / "runs")
    doc = memory.build_memory(led, runs)
    assert "| 量价相关 | 20 | dead |" in doc and "cc_run2" in doc
    assert "| 隔夜反转 | 20 | untested |" in doc
    assert "`rank(ts_corr(close, volume, 20))` | 20 | negative | -0.0300 | -2.40 | 是 |  |  | cc_run2" in doc
    assert "`ts_sum(returns, 5)` | 20 | positive | +0.0110 | +0.95 | 否 |  |  | cc_run1" in doc
    assert "run2 笔记" in doc and "截断" in doc
    for num in HIDDEN_NUMBERS:
        assert num not in doc, f"hidden tier number {num} leaked into the memory document"
    assert "PASS" not in doc and "holdout_ic" not in doc
    # the pre-registered one-bit feedback: b1 went through the full stage and was not adopted
    assert "| b1 | （无标签） | 20 | 未采纳 |" in doc
    outcomes = memory.full_stage_outcomes(led)
    assert outcomes == {"b1": {"adopted": False, "horizon": 20, "mechanism_tag": "",
                               "canonical": "rank(ts_corr(close, volume, 20))"}}
    assert len(doc) <= memory.MAX_CHARS + 200


def test_expression_index_only_carries_dev_columns(tmp_path):
    led = tmp_path / "mined.csv"; _ledger(led)
    idx = memory.expression_index(led)
    assert set(idx.columns) <= set(memory.DEV_COLUMNS)
    assert len(idx) == 2
    dup = memory.find_duplicate(led, "ts_sum(returns, 5)")
    assert dup and dup["candidate_id"] == "a1" and abs(dup["dev_t"] - 0.95) < 1e-9
    assert memory.find_duplicate(led, "rank(close)") is None
    assert memory.find_duplicate(tmp_path / "missing.csv", "rank(close)") is None


def test_screen_short_circuits_a_duplicate_expression(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", "2026-08-01", tz="America/New_York")
    n = len(dates)
    frames = []
    for i in range(45):
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({"date": dates, "symbol": f"S{i:03d}", "open": close, "high": close * 1.01,
                                    "low": close * 0.99, "close": close, "volume": np.exp(rng.normal(13, 0.2, n))}))
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    panel[harness.LABEL] = rng.normal(0, 0.05, len(panel))
    panel["pit"] = True
    ledger = tmp_path / "mined.csv"

    def prop(cid, expr):
        return Proposal(candidate_id=cid, expression=expr, expected_direction="positive",
                        hypothesis="A sentence long enough to validate.", mechanism="Another sentence long enough.",
                        refutation_conditions=["x"], source="test")

    first = harness.screen_one(prop("first", "ts_mean(returns, 5)"), panel, ledger)
    second = harness.screen_one(prop("second", "ts_mean(returns,5)"), panel, ledger)   # same canonical
    assert "duplicate_of" not in first
    assert second["duplicate_of"] == "first"
    assert abs(second["dev_t"] - first["dev_t"]) < 1e-12
    assert len(pd.read_csv(ledger)) == 1, "a duplicate must not add a ledger row"
    view = harness.agent_view(second)
    assert view["duplicate_of"] == "first" and "note" in view
