"""PreToolUse guard for the `miner` subagent (rulebook track B, step 3).

Claude Code runs this before EVERY tool call. For any caller that is not the
`miner` subagent it prints nothing and exits 0 (no effect). For the miner it
is a default-deny allowlist:

  Bash   only  <venv python> research/mining/harness.py [--horizon 5|20] ops | memory [--part N]
         and   <venv python> research/mining/harness.py screen
               research/mining/runs/cc_<run>/proposals[_N].json
               [--out research/mining/runs/cc_<run>/screen[_N].json]
  Write  only  research/mining/runs/cc_<run>/{proposals[_N].json, full_queue.json, notes.md, beliefs.json, reflection.md}
         and a proposals file is refused once the run's cumulative candidate
         count would exceed MAX_SCREENS (the budget is enforced here, not by
         the model's good will)
  everything else (Read, Grep, Glob, Edit, Agent, WebFetch, any other Bash)  DENY

The identity check keys on `agent_type`, which Claude Code sets in the hook
input; the model cannot set it. Every decision for the miner is appended to
research/mining/runs/guard.log for audit.

Pipe-test:  echo '{"agent_type":"miner","tool_name":"Read","tool_input":{}}' | python .claude/hooks/miner_guard.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # C:/Trading_System
RUNS = ROOT / "research" / "mining" / "runs"
MAX_SCREENS = 40

PY = r'(?:"?(?:C:/|/c/)Trading_System/research/venv/Scripts/python\.exe"?|\./?research/venv/Scripts/python\.exe|research/venv/Scripts/python\.exe)'
ROOTP = r"(?:(?:C:/|/c/)Trading_System/)?"          # optional absolute prefix, either spelling
HARNESS = rf"{ROOTP}research/mining/harness\.py"
RUN = rf"{ROOTP}research/mining/runs/cc_[A-Za-z0-9_]{{1,40}}"
HZ = r"(?:--horizon (?:5|20) )?"
BASH_OPS = re.compile(rf"^{PY} {HARNESS} {HZ}(ops|memory(?: --part [0-9]{{1,2}})?)$")
BASH_SCREEN = re.compile(
    rf"^{PY} {HARNESS} {HZ}screen ({RUN})/proposals(?:_\d{{1,3}})?\.json"
    rf"(?: --out \1/screen(?:_\d{{1,3}})?\.json)?$")
WRITE_OK = re.compile(r"^(proposals(?:_\d{1,3})?\.json|full_queue\.json|notes\.md|beliefs\.json|reflection\.md)$")


def _norm(p: str) -> Path:
    return Path(p.replace("\\", "/")).resolve()


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": f"miner guard: {reason}"}}


def _allow(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "allow",
                                   "permissionDecisionReason": f"miner guard: {reason}"}}


def _count_existing_candidates(run_dir: Path, exclude: Path) -> int:
    n = 0
    for f in run_dir.glob("proposals*.json"):
        if f.resolve() == exclude.resolve():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            items = data.get("proposals", data) if isinstance(data, dict) else data
            n += len(items) if isinstance(items, list) else 1
        except Exception:
            n += MAX_SCREENS          # unreadable file: assume the worst
    return n


def decide(data: dict) -> dict | None:
    if data.get("agent_type") != "miner":
        return None                    # not our agent: no opinion
    tool = data.get("tool_name", "")
    inp = data.get("tool_input") or {}

    if tool == "Bash":
        cmd = " ".join(str(inp.get("command", "")).split())
        if BASH_OPS.match(cmd):
            return _allow(cmd.rsplit(" ", 1)[-1])
        m = BASH_SCREEN.match(cmd)
        if m:
            return _allow("screen")
        return _deny("Bash is limited to `harness.py ops|memory` and `harness.py screen <run>/proposals*.json`")

    if tool == "Write":
        try:
            path = _norm(str(inp.get("file_path", "")))
        except Exception:
            return _deny("unreadable path")
        try:
            rel = path.relative_to(RUNS.resolve())
        except ValueError:
            return _deny("Write is limited to research/mining/runs/cc_<run>/")
        if len(rel.parts) != 2 or not re.match(r"^cc_[A-Za-z0-9_]{1,40}$", rel.parts[0]):
            return _deny("Write is limited to research/mining/runs/cc_<run>/<file>")
        if not WRITE_OK.match(rel.parts[1]):
            return _deny("only proposals*.json, full_queue.json, notes.md, beliefs.json and reflection.md may be written")
        if rel.parts[1].startswith("proposals"):
            content = str(inp.get("content", ""))
            try:
                body = json.loads(content)
                items = body.get("proposals", body) if isinstance(body, dict) else body
                new = len(items) if isinstance(items, list) else 1
            except Exception:
                return _deny("proposals file must be valid JSON")
            run_dir = RUNS / rel.parts[0]
            have = _count_existing_candidates(run_dir, run_dir / rel.parts[1]) if run_dir.exists() else 0
            if have + new > MAX_SCREENS:
                return _deny(f"budget: {have} already screened + {new} new > {MAX_SCREENS}")
        return _allow(f"write {rel.as_posix()}")

    return _deny(f"tool {tool} is not available to the miner")


def main() -> None:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    out = decide(data)
    if out is None:
        return
    try:
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(RUNS / "guard.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": datetime.now().isoformat(), "agent_id": data.get("agent_id"),
                                "tool": data.get("tool_name"), "input": data.get("tool_input"),
                                "decision": out["hookSpecificOutput"]["permissionDecision"],
                                "reason": out["hookSpecificOutput"]["permissionDecisionReason"]},
                               default=str) + "\n")
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
