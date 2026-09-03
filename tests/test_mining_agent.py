"""Pins the isolated mining agent (research/mining/agent.py) with the scripted
fake model: the loop runs offline, every tool result is dev-only, budgets and
queue rules hold, and the audit transcript is written."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import agent, dsl, harness  # noqa: E402

HIDDEN_WORDS = ("holdout", "virgin", "fresh", "verdict", "bh_threshold", "family_n")


def _panel(n_syms=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", "2026-08-01", tz="America/New_York")
    n = len(dates)
    frames = []
    for i in range(n_syms):
        close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i:03d}", "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": np.exp(rng.normal(13, 0.2, n))}))
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    sig = dsl.compile_expression("rank(ts_mean(returns, 5))", df)
    df[harness.LABEL] = 0.3 * (sig - 0.5) + rng.normal(0, 0.05, len(df))
    df["pit"] = True
    return df


def _proposal(cid, expr, direction="positive"):
    return {"candidate_id": cid, "expression": expr, "expected_direction": direction,
            "hypothesis": "Short-term return persistence predicts the next month.",
            "mechanism": "Under-reaction to recent news keeps prices drifting.",
            "refutation_conditions": ["sign flips out of sample"], "mechanism_tag": "persistence"}


SCRIPT = [
    [("list_operators", {})],
    [("screen_proposals", {"proposals": [
        _proposal("good_one", "rank(ts_mean(returns, 5))"),
        _proposal("wrong_way", "rank(ts_mean(returns, 5))", "negative"),
        _proposal("illegal", "close.shift(-1)")]})],
    [("request_full_stage", {"candidate_id": "good_one"}),
     ("request_full_stage", {"candidate_id": "wrong_way"}),
     ("request_full_stage", {"candidate_id": "never_screened"})],
    [("screen_proposals", {"proposals": [_proposal("over_budget", "rank(close)")]})],
    [("submit_notes", {"notes": "test notes", "beliefs": [
        {"mechanism": "persistence", "status": "promising", "evidence": "t>2", "next": "vary window"}]})],
]


def test_tools_are_strict_and_closed():
    for t in agent.TOOLS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False
    names = {t["name"] for t in agent.TOOLS}
    assert names == {"read_memory", "list_operators", "screen_proposals", "request_full_stage", "submit_notes"}


def test_system_prompt_states_budget_and_incumbents():
    s = agent.system_prompt("find things", 12)
    assert "12" in s and "high52" in s and "find things" in s
    assert "development-segment" in s and "read_memory" in s


def test_dry_loop_is_isolated_and_audited(tmp_path):
    panel = _panel()
    run_dir = tmp_path / "run"
    ledger = tmp_path / "mined.csv"
    sb = agent.Sandbox(run_dir, panel, max_screens=3, ledger_path=ledger, record=True)
    client = agent.FakeClient([list(step) for step in SCRIPT])
    summary = agent.run_session(client, sb, "goal", model="fake", max_turns=10, fallbacks=False)

    assert summary["stop"] == "end_turn"
    assert summary["screened"] == 3
    assert summary["passed"] == ["good_one"]
    assert summary["queued_full"] == ["good_one"]
    assert summary["notes"] is True

    # the only thing the model ever received: tool results
    lines = [json.loads(l) for l in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "system" and kinds[-1] == "summary"
    tool_results = [l["payload"]["result"] for l in lines if l["kind"] == "tool"]
    assert len(tool_results) == 7          # 1 ops + 2 screens + 3 queue calls + 1 notes
    for text in tool_results:
        low = text.lower()
        for w in HIDDEN_WORDS:
            assert w not in low, f"hidden word {w!r} reached a tool result"

    views = {v["candidate_id"]: v for v in sb.screened.values()}
    assert views["good_one"]["screen_pass"] is True
    assert views["wrong_way"]["screen_pass"] is False
    assert "expression:" in views["illegal"]["error"]

    # queue rules
    queue_results = [json.loads(l["payload"]["result"]) for l in lines
                     if l["kind"] == "tool" and l["payload"]["name"] == "request_full_stage"]
    assert [q["queued"] for q in queue_results] == [True, False, False]
    # budget: the fourth candidate is refused
    over = [json.loads(l["payload"]["result"]) for l in lines
            if l["kind"] == "tool" and l["payload"]["name"] == "screen_proposals"][1]
    assert "budget" in over[0]["error"]

    # artefacts for the human and for --run-full
    assert json.loads((run_dir / "full_queue.json").read_text()) == ["good_one"]
    props = json.loads((run_dir / "proposals.json").read_text())["proposals"]
    assert {p["candidate_id"] for p in props} == {"good_one", "wrong_way", "illegal"}
    assert (run_dir / "notes.md").read_text(encoding="utf-8") == "test notes"
    beliefs = json.loads((run_dir / "beliefs.json").read_text(encoding="utf-8"))
    assert beliefs["beliefs"][0]["mechanism"] == "persistence" and beliefs["run_id"] == "run"
    assert views["wrong_way"]["duplicate_of"] == "good_one"     # same expression, memory hit
    led = pd.read_csv(ledger)
    assert led["stage"].tolist() == ["screen"] * 2 and led["source"].str.startswith("agent:").all()


def test_loop_stops_at_max_turns(tmp_path):
    panel = _panel(n_syms=40)
    sb = agent.Sandbox(tmp_path / "r", panel, max_screens=5, ledger_path=tmp_path / "l.csv", record=False)
    endless = [[("list_operators", {})] for _ in range(20)]
    client = agent.FakeClient(endless)
    summary = agent.run_session(client, sb, "goal", model="fake", max_turns=3, fallbacks=False)
    assert summary["stop"] == "max_turns" and summary["usage"]["turns"] == 3
