"""Pins the segment-rotation review (research/mining/harness.review): the
segments of --date are applied at runtime only and restored afterwards, every
full-stage candidate of the horizon is re-adjudicated once, the whole set is
one BH family (pass 2), nothing is appended to the ledger, and the report
files are written."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from evaluation import rulebook as rb  # noqa: E402
from mining import harness  # noqa: E402


def _ledger(path: Path) -> Path:
    rows = []
    for cid, expr, d in (("a_one", "rank(close)", "positive"), ("b_two", "rank(volume)", "negative")):
        rows.append({"date": "2026-09-01", "candidate_id": cid, "proposal_hash": cid, "source": "cc-miner:x",
                     "expected_direction": d, "stage": "full", "expression": expr, "canonical": expr,
                     "verdict": "FAIL", "horizon": 20, "mechanism_tag": "m_" + cid,
                     "pooled_p_onesided": 0.5, "recent_p_onesided": 0.6})
    rows.append({**rows[0], "stage": "screen", "verdict": "screen_pass"})
    rows.append({**rows[0], "horizon": 5, "candidate_id": "h5_only"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fake_adjudicate(p, tag, results_dir=None, family=None, ledger_path=None, raw_feature=None):
    strong = p.candidate_id == "a_one"
    sign = 1.0 if p.expected_direction == "positive" else -1.0
    t = 3.5 if strong else 0.3
    row = {"candidate_id": p.candidate_id, "expected_direction": p.expected_direction, "expression": p.expression,
           "dev_ic": sign * 0.02, "dev_t": sign * 3.0, "dev_n": 900, "blend_dev_gain": 0.002,
           "virgin_ic": sign * 0.01, "virgin_t": sign * t, "virgin_n": 1000,
           "holdout_ic": sign * 0.02, "holdout_t": sign * t, "holdout_n": 250,
           "fresh_ic": 0.0, "fresh_t": 0.0, "fresh_n": 0,
           "pooled_ic": sign * 0.012, "pooled_t": sign * t, "pooled_n": 1250,
           "recent_ic": sign * 0.02, "recent_t": sign * t, "recent_n": 250}
    row["pooled_p_onesided"] = rb.onesided_p(row["pooled_t"], p.expected_direction)
    row["recent_p_onesided"] = rb.onesided_p(row["recent_t"], p.expected_direction)
    row["holdout_p_onesided"] = rb.onesided_p(row["holdout_t"], p.expected_direction)
    row.update({"verdict": "?", "reasons": "", "family_n": 0, "bh_threshold": 0.0, "adoption_tier": ""})
    return row


def test_review_rotates_at_runtime_and_bh_over_the_whole_set(tmp_path):
    ledger = _ledger(tmp_path / "ledger.csv")
    harness.configure(20)
    before = harness.SEGMENTS
    n_rows_before = len(pd.read_csv(ledger))
    out = harness.review("2026-11-15", rerun=False, ledger_path=ledger, results_dir=tmp_path, adjudicate=_fake_adjudicate)
    assert harness.SEGMENTS == before                                   # restored
    assert out["n_candidates"] == 2 and out["horizon"] == 20             # h=5 row and screen row ignored
    seg = dict((n, (s, e)) for n, s, e in out["segments"])
    assert seg["holdout"] == ("2025-11-15", "2026-11-15") and seg["fresh"] == ("2026-11-15", None)
    by = {r["candidate_id"]: r for r in out["rows"]}
    assert by["a_one"]["verdict"] == "PASS" and by["a_one"]["adoption_tier"] == "structural"
    assert by["b_two"]["verdict"] == "FAIL"
    assert by["a_one"]["family_n"] == 4                                  # own 2 + the other's 2
    assert by["a_one"]["previous_verdict"] == "FAIL"
    assert len(pd.read_csv(ledger)) == n_rows_before                     # ledger untouched
    assert (tmp_path / "review_2026-11-15_h20.json").exists()
    md = (tmp_path / "review_2026-11-15_h20.md").read_text(encoding="utf-8")
    assert "a_one" in md and "**PASS**" in md and "not an adoption" in md
    assert json.loads((tmp_path / "review_2026-11-15_h20.json").read_text(encoding="utf-8"))["passes"] == ["a_one"]
