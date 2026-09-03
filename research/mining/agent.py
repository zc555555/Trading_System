"""Isolated mining agent (step 3 of the plan; rulebook track B).

    python mining/agent.py --goal "..." [--model claude-opus-5] [--max-screens 40]
    python mining/agent.py --dry-run          # scripted fake model, no API call
    python mining/agent.py --run-full RUN_ID  # run queued full stages of a finished run

Isolation is technical, not instructional. The language model runs in this
process with exactly four client-side tools, all implemented here:

    list_operators()               the DSL registry text
    read_memory()                  cross-run research memory (dev-only, see memory.py)
    screen_proposals(proposals)    stage-1 screen -> agent_view() of each record
    request_full_stage(id)         enqueue a screen-passing candidate; runs later
    submit_notes(text)             end-of-run research notes (input to step 4)

It has no file, shell, or network tool. Everything it learns arrives as a
tool_result built by this file, and every tool_result passes through
`harness.agent_view()`, which drops the hidden tiers. The full stage is never
run inside the session: the candidate is queued and the human (or
`--run-full`) runs it afterwards, so holdout/virgin/fresh numbers cannot even
exist in the process while the model is talking. Every request, tool call
and tool result is appended to runs/<run_id>/transcript.jsonl for audit.

Model: claude-opus-5 by default (adaptive thinking is on by default there;
`output_config.effort` controls depth). Server-side refusal fallbacks are
enabled by default (`fallbacks="default"`); pass --no-fallbacks to disable.
Credentials: ANTHROPIC_API_KEY in the environment or in config_keys.py
(same convention as the data-vendor keys); otherwise the SDK's own
resolution (an `ant auth login` profile) applies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from mining import dsl, harness                       # noqa: E402
from mining.proposal import Proposal                  # noqa: E402
from evaluation import rulebook as rb                 # noqa: E402

RUNS = RESEARCH / "mining" / "runs"
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

INCUMBENTS = (
    "momentum (multi-horizon returns, RSI/MACD/CCI oscillators), trend (moving averages, ADX, "
    "channels), volatility (realised vol, ATR, Bollinger width), volume (volume ratios, OBV, MFI), "
    "market (index/VIX context), alpha (a 101-alphas subset), high52 (close / 52-week high)"
)

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string", "description": "^[a-z][a-z0-9_]{2,30}$, unique"},
        "expression": {"type": "string", "description": "one line in the DSL"},
        "expected_direction": {"type": "string", "enum": ["positive", "negative"]},
        "hypothesis": {"type": "string"},
        "mechanism": {"type": "string"},
        "refutation_conditions": {"type": "array", "items": {"type": "string"}},
        "mechanism_tag": {"type": "string", "description": "short stable mechanism name, matches beliefs"},
    },
    "required": ["candidate_id", "expression", "expected_direction", "hypothesis",
                 "mechanism", "refutation_conditions", "mechanism_tag"],
    "additionalProperties": False,
}

TOOLS = [
    {"name": "read_memory", "strict": True,
     "description": ("Return the cross-run research memory: mechanism statuses from earlier runs' "
                     "beliefs, the index of every expression already screened (with its dev statistics; "
                     "duplicates are not re-scored), and the last runs' notes. Development-segment only."),
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "list_operators", "strict": True,
     "description": "Return the operator registry, field list, syntax and size limits of the factor DSL.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "screen_proposals", "strict": True,
     "description": ("Stage-1 screen of up to 8 proposals: standalone per-date rank IC of each expression "
                     "against the 20-session forward return on the development segment only. Returns, per "
                     "candidate, dev_ic, dev_t (Newey-West, 19 lags), dev_n, coverage, lookback, n_nodes and "
                     "screen_pass (signed dev_t >= 2.0 in the declared direction and coverage >= 0.90). "
                     "Nothing outside the development segment is ever returned."),
     "input_schema": {"type": "object",
                      "properties": {"proposals": {"type": "array", "items": PROPOSAL_SCHEMA,
                                                   "minItems": 1, "maxItems": 8}},
                      "required": ["proposals"], "additionalProperties": False}},
    {"name": "request_full_stage", "strict": True,
     "description": ("Queue a candidate that passed the screen for the full purged walk-forward. The full "
                     "stage runs after this session ends (about 35 minutes each) and its hidden tiers are "
                     "adjudicated by a human; you will not see its result in this session."),
     "input_schema": {"type": "object", "properties": {"candidate_id": {"type": "string"}},
                      "required": ["candidate_id"], "additionalProperties": False}},
    {"name": "submit_notes", "strict": True,
     "description": ("Record your end-of-run research notes (free text) and structured beliefs: one entry "
                     "per mechanism with status dead|weak|promising|untested, the evidence, and what to try "
                     "next. Later runs read these. Call once, at the end."),
     "input_schema": {"type": "object",
                      "properties": {"notes": {"type": "string"},
                                     "beliefs": {"type": "array", "items": {
                                         "type": "object",
                                         "properties": {"mechanism": {"type": "string"},
                                                        "status": {"type": "string", "enum": ["dead", "weak", "promising", "untested"]},
                                                        "evidence": {"type": "string"}, "next": {"type": "string"}},
                                         "required": ["mechanism", "status", "evidence", "next"],
                                         "additionalProperties": False}}},
                      "required": ["notes", "beliefs"], "additionalProperties": False}},
]


def system_prompt(goal: str, max_screens: int) -> str:
    return f"""You are the factor-mining agent of a small, honestly evaluated US-equity research system.

Your job: propose cross-sectional stock-selection factors as one-line expressions in a fixed operator
DSL, each wrapped in a falsifiable proposal (hypothesis, economic mechanism, expected direction,
refutation conditions), and screen them. You are a proposer inside a sealed sandbox:
- The data, label (20-session forward log return), splits and evaluator are fixed and not visible to you.
- You only ever see the development-segment score of your own candidates. Other evidence tiers exist,
  are computed by the harness, and are adjudicated by a human after this session. Do not ask for them.
- Declare expected_direction before screening; a candidate whose later hidden-tier sign disagrees with
  its declaration fails automatically, so do not hedge with both signs of the same expression.
- Universe: point-in-time S&P 500 members including delisted names, daily bars, 2015-2026.
  Honest single-factor rank ICs here are small (0.01-0.03); a screen pass needs signed dev t >= 2.0.

Incumbent production factors (do not restate them): {INCUMBENTS}.

Research goal for this run: {goal}

Method: call read_memory first (what earlier runs learned; do not re-screen indexed expressions or
dead mechanisms), then list_operators. Then work in batches of up to 8 proposals per screen_proposals
call, reading the results before the next batch: keep mechanisms that screen well and vary their
window, normalisation or conditioning; drop mechanisms that screen badly rather than re-parametrising
them many times. Prefer expressions with a stated economic reason over syntactic novelty. Budget:
at most {max_screens} candidates this run. For each screen pass, decide whether to call
request_full_stage (only strong, mechanism-grounded passes; full stages are expensive). Finish with
one submit_notes call, then stop."""


# --------------------------------------------------------------------------
# the four tools (the model's entire world)
# --------------------------------------------------------------------------
class Sandbox:
    def __init__(self, run_dir: Path, panel: pd.DataFrame, max_screens: int,
                 ledger_path: Path = rb.MINED_LEDGER, record: bool = True):
        self.run_dir = run_dir
        self.panel = panel
        self.max_screens = max_screens
        self.ledger_path = ledger_path
        self.record = record
        self.screened: dict[str, dict] = {}
        self.queued: list[str] = []
        self.notes: str | None = None
        self.goal: str = ""
        self.proposals: dict[str, dict] = {}
        run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_screened(self) -> int:
        return len(self.screened)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Dispatch. Returns (content, is_error)."""
        try:
            if name == "list_operators":
                return dsl.describe_ops(), False
            if name == "read_memory":
                from mining.memory import build_memory
                return build_memory(self.ledger_path, RUNS), False
            if name == "screen_proposals":
                return json.dumps(self.screen(args["proposals"]), indent=1), False
            if name == "request_full_stage":
                return json.dumps(self.request_full(args["candidate_id"])), False
            if name == "submit_notes":
                self.notes = str(args["notes"])
                (self.run_dir / "notes.md").write_text(self.notes, encoding="utf-8")
                beliefs = {"run_id": self.run_dir.name, "goal": self.goal,
                           "beliefs": list(args.get("beliefs") or [])}
                (self.run_dir / "beliefs.json").write_text(json.dumps(beliefs, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
                return json.dumps({"recorded": True, "beliefs": len(beliefs["beliefs"])}), False
            return f"unknown tool {name}", True
        except Exception as e:                       # never leak a traceback with paths
            return f"{type(e).__name__}: {e}", True

    def screen(self, items: list[dict]) -> list[dict]:
        out = []
        for item in items:
            cid = str(item.get("candidate_id", ""))
            if cid in self.screened or cid in self.proposals:
                out.append({"candidate_id": cid, "error": "candidate_id already used in this run"})
                continue
            if self.n_screened >= self.max_screens:
                out.append({"candidate_id": cid, "error": f"budget of {self.max_screens} candidates exhausted"})
                continue
            p = Proposal(candidate_id=cid, expression=str(item.get("expression", "")),
                         expected_direction=str(item.get("expected_direction", "")),
                         hypothesis=str(item.get("hypothesis", "")),
                         mechanism=str(item.get("mechanism", "")),
                         refutation_conditions=[str(c) for c in item.get("refutation_conditions", [])],
                         mechanism_tag=str(item.get("mechanism_tag", "")),
                         source=f"agent:{self.run_dir.name}")
            self.proposals[cid] = p.to_dict()
            row = harness.screen_one(p, self.panel, self.ledger_path, record=self.record)
            view = harness.agent_view(row)
            assert not any(k.startswith(("holdout", "virgin", "fresh")) or k in ("verdict", "reasons")
                           for k in view), "hidden field reached the agent view"
            self.screened[cid] = view
            out.append(view)
        (self.run_dir / "proposals.json").write_text(
            json.dumps({"proposals": list(self.proposals.values())}, indent=1), encoding="utf-8")
        return out

    def request_full(self, cid: str) -> dict:
        v = self.screened.get(cid)
        if v is None:
            return {"candidate_id": cid, "queued": False, "reason": "not screened in this run"}
        if not v.get("screen_pass"):
            return {"candidate_id": cid, "queued": False, "reason": "did not pass the screen"}
        if cid not in self.queued:
            self.queued.append(cid)
            (self.run_dir / "full_queue.json").write_text(json.dumps(self.queued), encoding="utf-8")
        return {"candidate_id": cid, "queued": True,
                "note": "runs after this session; result adjudicated by a human"}


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def _log(run_dir: Path, kind: str, payload) -> None:
    with open(run_dir / "transcript.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.now().isoformat(), "kind": kind, "payload": payload},
                           default=str) + "\n")


def _serialise_content(content) -> list:
    """Turn response content blocks into plain dicts the API accepts back."""
    out = []
    for b in content:
        if hasattr(b, "model_dump"):
            out.append(b.model_dump(exclude_none=True))
        elif isinstance(b, dict):
            out.append(b)
        else:
            out.append({k: v for k, v in vars(b).items() if not k.startswith("_")})
    return out


def run_session(client, sandbox: Sandbox, goal: str, model: str = DEFAULT_MODEL,
                max_turns: int = 40, effort: str = "high", fallbacks: bool = True) -> dict:
    """Manual tool loop: one request per turn, all tool results in one user
    message, hard cap on turns. Returns a summary."""
    run_dir = sandbox.run_dir
    sandbox.goal = goal
    system = system_prompt(goal, sandbox.max_screens)
    messages = [{"role": "user", "content": "Begin. Call read_memory, then list_operators."}]
    _log(run_dir, "system", {"model": model, "goal": goal, "system": system})

    usage = {"input": 0, "output": 0, "cache_read": 0, "turns": 0}
    stop = "max_turns"
    for turn in range(max_turns):
        kwargs = dict(model=model, max_tokens=MAX_TOKENS,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                      tools=TOOLS, messages=messages,
                      thinking={"type": "adaptive"}, output_config={"effort": effort})
        if fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            response = client.beta.messages.create(**kwargs)
        else:
            response = client.messages.create(**kwargs)
        usage["turns"] += 1
        u = getattr(response, "usage", None)
        if u is not None:
            usage["input"] += int(getattr(u, "input_tokens", 0) or 0)
            usage["output"] += int(getattr(u, "output_tokens", 0) or 0)
            usage["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        content = _serialise_content(response.content)
        _log(run_dir, "assistant", {"turn": turn, "stop_reason": response.stop_reason, "content": content})

        if response.stop_reason == "refusal":
            stop = "refusal"
            break
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            stop = response.stop_reason or "end_turn"
            break

        messages.append({"role": "assistant", "content": content})
        results = []
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
            text, is_error = sandbox.call(tu.name, args)
            _log(run_dir, "tool", {"turn": turn, "name": tu.name, "input": args,
                                   "is_error": is_error, "result": text})
            block = {"type": "tool_result", "tool_use_id": tu.id, "content": text}
            if is_error:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})

    summary = {"run_id": run_dir.name, "model": model, "stop": stop, "usage": usage,
               "screened": sandbox.n_screened,
               "passed": [c for c, v in sandbox.screened.items() if v.get("screen_pass")],
               "queued_full": list(sandbox.queued), "notes": bool(sandbox.notes)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _log(run_dir, "summary", summary)
    return summary


# --------------------------------------------------------------------------
# dry run: a scripted fake model, so the isolation loop is testable offline
# --------------------------------------------------------------------------
class FakeClient:
    """Emits a fixed script of tool calls; mimics the SDK response shape."""

    def __init__(self, script: list[list[tuple[str, dict]]]):
        self.script = list(script)
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else None
        if not step:
            return SimpleNamespace(stop_reason="end_turn", usage=None,
                                   content=[SimpleNamespace(type="text", text="done")])
        blocks = [SimpleNamespace(type="tool_use", id=f"toolu_{self.calls}_{i}", name=n, input=a)
                  for i, (n, a) in enumerate(step)]
        return SimpleNamespace(stop_reason="tool_use", usage=None, content=blocks)


DRY_SCRIPT = [
    [("list_operators", {})],
    [("screen_proposals", {"proposals": [
        {"candidate_id": "dry_reversal", "expression": "neg(ts_sum(returns, 5))",
         "expected_direction": "positive",
         "hypothesis": "Last week's losers outperform over the next month.",
         "mechanism": "Liquidity provision: temporary price pressure reverts.",
         "refutation_conditions": ["dev IC not positive"]},
        {"candidate_id": "dry_bad", "expression": "close.shift(-1)", "expected_direction": "positive",
         "hypothesis": "This expression is illegal and must be rejected by the DSL.",
         "mechanism": "None; it tests that leakage cannot be written.",
         "refutation_conditions": ["any pass"]}]})],
    [("request_full_stage", {"candidate_id": "dry_reversal"}),
     ("submit_notes", {"notes": "dry run complete", "beliefs": [
         {"mechanism": "short-term reversal", "status": "weak", "evidence": "dry", "next": "none"}]})],
]


def make_client(model: str):
    import anthropic
    from data.vendor_keys import get_key
    key = get_key("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def run_full_queue(run_id: str) -> list[dict]:
    """Human/driver: run the full stage for every candidate the agent queued."""
    run_dir = RUNS / run_id
    queue = json.loads((run_dir / "full_queue.json").read_text(encoding="utf-8"))
    # API runs write one proposals.json; Claude Code miner runs (cc_*) write
    # proposals_1.json, proposals_2.json, ... -- merge every batch.
    props = {}
    for f in sorted(run_dir.glob("proposals*.json")):
        body = json.loads(f.read_text(encoding="utf-8"))
        for p in body.get("proposals", body) if isinstance(body, dict) else body:
            props[p["candidate_id"]] = Proposal(
                **{k: v for k, v in p.items() if k in Proposal.__dataclass_fields__})
    missing = [c for c in queue if c not in props]
    if missing:
        raise SystemExit(f"queued ids not found in any proposals file: {missing}")
    # the full stage runs at the run's own horizon (one track-B family per horizon)
    horizons = {int(p.horizon) for p in props.values()}
    if len(horizons) != 1:
        raise SystemExit(f"run {run_id} mixes horizons {sorted(horizons)}; refusing")
    harness.configure(horizons.pop())
    print(f"full stages at horizon {harness.HORIZON} (label {harness.LABEL}, baseline "
          f"oos_predictions_h{harness.HORIZON}_{harness.BASELINE_TAG}.parquet)")
    out = []
    for cid in queue:
        t0 = time.time()
        try:
            row = harness.full_one(props[cid])
        except SystemExit as e:                     # non-representative: skip, say why
            view = {"candidate_id": cid, "skipped": str(e)}
            out.append(view); _log(run_dir, "full_stage", view); print(json.dumps(view, indent=1))
            continue
        view = harness.agent_view(row)
        view["elapsed_s"] = round(time.time() - t0)
        out.append(view)
        _log(run_dir, "full_stage", view)          # dev-only view in the transcript
        print(json.dumps(view, indent=1))
    (run_dir / "full_results_agent_view.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", default="Find price-volume mechanisms not covered by the incumbent factors.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-screens", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not write the mined ledger (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="scripted fake model, no API call")
    ap.add_argument("--run-full", metavar="RUN_ID", help="run the queued full stages of a finished run")
    args = ap.parse_args(argv)

    if args.run_full:
        run_full_queue(args.run_full)
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dry" if args.dry_run else "")
    run_dir = RUNS / run_id
    panel = harness.load_screen_panel()
    sandbox = Sandbox(run_dir, panel, args.max_screens, record=not (args.no_record or args.dry_run))
    client = FakeClient(DRY_SCRIPT) if args.dry_run else make_client(args.model)
    summary = run_session(client, sandbox, args.goal, model=args.model, max_turns=args.max_turns,
                          effort=args.effort, fallbacks=not args.no_fallbacks)
    print(json.dumps(summary, indent=1))
    if sandbox.queued:
        print(f"\nqueued full stages: python mining/agent.py --run-full {run_id}")


if __name__ == "__main__":
    main()
