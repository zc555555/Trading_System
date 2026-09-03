"""Pins the Claude Code PreToolUse guard for the `miner` subagent
(.claude/hooks/miner_guard.py): no opinion for other callers, default deny
for the miner, the two allowed commands, the three allowed files, and the
candidate budget."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / ".claude" / "hooks" / "miner_guard.py"

spec = importlib.util.spec_from_file_location("miner_guard", GUARD)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

PY = "research/venv/Scripts/python.exe"
RUN = "research/mining/runs/cc_test"


def _bash(cmd, agent="miner"):
    return guard.decide({"agent_type": agent, "tool_name": "Bash", "tool_input": {"command": cmd}})


def _write(path, content='{"proposals": []}', agent="miner"):
    return guard.decide({"agent_type": agent, "tool_name": "Write",
                         "tool_input": {"file_path": str(ROOT / path), "content": content}})


def _dec(out):
    return None if out is None else out["hookSpecificOutput"]["permissionDecision"]


def test_no_opinion_for_anyone_but_the_miner():
    assert guard.decide({"tool_name": "Bash", "tool_input": {"command": "cat secrets"}}) is None
    assert _bash("cat research/evaluation/results/mined_candidates.csv", agent="general-purpose") is None
    assert guard.decide({"agent_type": "Explore", "tool_name": "Read", "tool_input": {}}) is None


def test_miner_bash_allowlist():
    assert _dec(_bash(f"{PY} research/mining/harness.py ops")) == "allow"
    assert _dec(_bash(f"{PY} research/mining/harness.py memory")) == "allow"
    assert _dec(_bash(f"{PY} research/mining/harness.py memory --anything")) == "deny"
    assert _dec(_bash(f"{PY} research/mining/harness.py screen {RUN}/proposals_1.json")) == "allow"
    assert _dec(_bash(f"{PY} research/mining/harness.py screen {RUN}/proposals.json --out {RUN}/screen.json")) == "allow"
    assert _dec(_bash(f'"C:/Trading_System/research/venv/Scripts/python.exe" research/mining/harness.py ops')) == "allow"
    for bad in [
        f"{PY} research/mining/harness.py show cand",
        f"{PY} research/mining/harness.py full {RUN}/proposals_1.json --id x",
        f"{PY} research/mining/harness.py adopt x --removal-trigger t",
        f"{PY} research/mining/harness.py screen {RUN}/proposals_1.json; cat research/evaluation/results/mined_candidates.csv",
        f"{PY} research/mining/harness.py screen {RUN}/proposals_1.json && ls",
        f"{PY} research/mining/harness.py screen {RUN}/proposals_1.json | tee x",
        f"{PY} research/mining/harness.py screen ../../evaluation/results/mined_candidates.csv",
        f"{PY} research/mining/harness.py screen research/mining/runs/other/proposals_1.json",
        f"{PY} research/mining/harness.py screen {RUN}/proposals_1.json --out research/evaluation/results/x.json",
        f"{PY} research/mining/agent.py --dry-run",
        f"{PY} -c \"print(open('research/evaluation/results/mined_candidates.csv').read())\"",
        "cat research/evaluation/results/mined_candidates.csv",
        "python research/mining/harness.py ops",
        "ls", "true", "",
    ]:
        assert _dec(_bash(bad)) == "deny", bad


def test_miner_write_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "RUNS", tmp_path / "runs")
    base = tmp_path / "runs" / "cc_x"
    for name in ("proposals.json", "proposals_2.json", "full_queue.json", "notes.md", "beliefs.json", "reflection.md"):
        out = guard.decide({"agent_type": "miner", "tool_name": "Write",
                            "tool_input": {"file_path": str(base / name),
                                           "content": '{"proposals": []}' if name.startswith("proposals") else "[]"}})
        assert _dec(out) == "allow", name
    for bad in (tmp_path / "runs" / "cc_x" / "other.txt",
                tmp_path / "runs" / "cc_x" / "sub" / "proposals.json",
                tmp_path / "runs" / "nope" / "proposals.json",
                tmp_path / "elsewhere" / "proposals.json",
                ROOT / "research" / "factors" / "mined_factors.json",
                ROOT / ".claude" / "hooks" / "miner_guard.py"):
        out = guard.decide({"agent_type": "miner", "tool_name": "Write",
                            "tool_input": {"file_path": str(bad), "content": "{}"}})
        assert _dec(out) == "deny", bad


def test_miner_budget_is_enforced_at_write(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(guard, "MAX_SCREENS", 10)
    run = tmp_path / "runs" / "cc_b"
    run.mkdir(parents=True)
    (run / "proposals_1.json").write_text(json.dumps({"proposals": [{"candidate_id": f"c{i}"} for i in range(8)]}))
    ok = json.dumps({"proposals": [{"candidate_id": "d1"}, {"candidate_id": "d2"}]})
    over = json.dumps({"proposals": [{"candidate_id": "d1"}, {"candidate_id": "d2"}, {"candidate_id": "d3"}]})
    assert _dec(guard.decide({"agent_type": "miner", "tool_name": "Write",
                              "tool_input": {"file_path": str(run / "proposals_2.json"), "content": ok}})) == "allow"
    assert _dec(guard.decide({"agent_type": "miner", "tool_name": "Write",
                              "tool_input": {"file_path": str(run / "proposals_2.json"), "content": over}})) == "deny"
    assert _dec(guard.decide({"agent_type": "miner", "tool_name": "Write",
                              "tool_input": {"file_path": str(run / "proposals_3.json"), "content": "not json"}})) == "deny"


def test_every_other_tool_is_denied_for_the_miner():
    for tool in ("Read", "Grep", "Glob", "Edit", "Agent", "WebFetch", "WebSearch", "PowerShell", "NotebookEdit"):
        assert _dec(guard.decide({"agent_type": "miner", "tool_name": tool, "tool_input": {}})) == "deny", tool


def test_guard_runs_as_a_process_and_emits_hook_json():
    payload = json.dumps({"agent_type": "miner", "agent_id": "t", "tool_name": "Read", "tool_input": {}})
    r = subprocess.run([sys.executable, str(GUARD)], input=payload, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    r2 = subprocess.run([sys.executable, str(GUARD)], input='{"tool_name":"Bash","tool_input":{"command":"ls"}}',
                        capture_output=True, text=True, timeout=30)
    assert r2.returncode == 0 and r2.stdout.strip() == ""
