"""Cross-run research memory for the mining agent (step 4; AQuA's Research
Librarian + beliefs store, on our terms).

The miner cannot read files. Its memory is a document this module builds
from three sources and the harness prints via `harness.py memory`:

  1. beliefs.json written by the agent at the end of each run:
        {"run_id": ..., "goal": ..., "beliefs": [
            {"mechanism": ..., "status": "dead|weak|promising|untested",
             "evidence": ..., "next": ...}, ...]}
     merged across runs by mechanism (the latest run's entry wins);
  2. the mined ledger's SCREEN rows: an index of every canonical expression
     already screened, with its declared direction and dev statistics, so a
     duplicate is never re-screened (harness.screen_one returns the old
     record instead);
  3. the last few runs' notes.md, verbatim, truncated.

What never enters the memory: any hidden tier (virgin_early, holdout,
fresh), any verdict, the BH family. Only dev-stage columns are read from
the ledger, by name. A full-stage row contributes nothing beyond its dev
fields. Adopted factors become incumbents and are visible that way, which
is unavoidable and harmless: production is public to the agent by
construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent
RUNS = RESEARCH / "mining" / "runs"
MAX_CHARS = 60000        # 2026-09-04: 16000 truncated the expression index after twelve runs
NOTES_PER_RUN = 5000
RECENT_RUNS = 2
STATUSES = ("dead", "weak", "promising", "untested")
DEV_COLUMNS = ["date", "candidate_id", "source", "expected_direction", "canonical",
               "expression", "dev_ic", "dev_t", "coverage", "screen_pass", "stage", "mechanism_tag",
               "cluster_rep", "redundant_with", "residual_dev_t", "horizon", "universe"]


# --------------------------------------------------------------------------
def _run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.endswith("_dry")]
    return sorted(dirs, key=lambda d: d.stat().st_mtime)


def load_beliefs(runs_dir: Path = RUNS) -> tuple[dict, list[str]]:
    """Merge beliefs across runs; later runs overwrite earlier ones per
    (mechanism, horizon). Returns (key -> belief dict with run_id and horizon,
    ordered run ids). A run's horizon is its beliefs.json "horizon" (20 if absent)."""
    merged: dict[str, dict] = {}
    seen_runs = []
    for d in _run_dirs(runs_dir):
        f = d / "beliefs.json"
        if not f.exists():
            continue
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(body.get("run_id") or d.name)
        run_h = int(body.get("horizon") or 20)
        seen_runs.append(run_id)
        for b in body.get("beliefs", []):
            mech = str(b.get("mechanism", "")).strip()
            if not mech:
                continue
            status = str(b.get("status", "untested")).strip().lower()
            h = int(b.get("horizon") or run_h)
            merged[(mech.lower(), h)] = {
                "mechanism": mech, "horizon": h,
                "status": status if status in STATUSES else "untested",
                "evidence": str(b.get("evidence", "")).strip(),
                "next": str(b.get("next", "")).strip(),
                "run_id": run_id,
            }
    return merged, seen_runs


def expression_index(ledger_path: Path) -> pd.DataFrame:
    """Every canonical expression ever screened, dev fields only, latest row
    per canonical form."""
    if not Path(ledger_path).exists():
        return pd.DataFrame(columns=DEV_COLUMNS)
    from evaluation import rulebook as rb
    led = rb.load_mined_ledger(ledger_path)
    cols = [c for c in DEV_COLUMNS if c in led.columns]
    led = led[cols]                                    # hidden columns never loaded further
    scr = led[led["stage"].astype(str) == "screen"].copy()
    scr = scr[scr["canonical"].notna()]
    scr = scr.sort_values("date").drop_duplicates("canonical", keep="last")
    return scr.reset_index(drop=True)


def find_duplicate(ledger_path: Path, canonical: str, horizon: int | None = None,
                   universe: str | None = None) -> dict | None:
    """The same expression at a DIFFERENT horizon or universe is a different test."""
    idx = expression_index(ledger_path)
    hit = idx[idx["canonical"] == canonical]
    from evaluation import rulebook as rb
    if horizon is not None and len(hit) and "horizon" in hit.columns:
        hit = hit[rb.horizon_of(hit) == int(horizon)]
    if universe is not None and len(hit):
        hit = hit[rb.universe_of(hit) == universe]
    if hit.empty:
        return None
    r = hit.iloc[-1]
    return {k: (None if pd.isna(r[k]) else r[k]) for k in hit.columns}


def full_stage_outcomes(ledger_path: Path, horizon: int | None = None) -> dict[str, dict]:
    """The pre-registered one-bit feedback (RULEBOOK, 2026-09-03): for every
    candidate that went through the full stage, whether it was adopted --
    and nothing else. Reads only candidate_id / stage / mechanism_tag /
    canonical from the ledger; no tier, no verdict, no number."""
    if not Path(ledger_path).exists():
        return {}
    from evaluation import rulebook as rb
    led = rb.load_mined_ledger(ledger_path)
    if led.empty:
        return {}
    from evaluation import rulebook as rb
    led = led.assign(_h=rb.horizon_of(led))
    if horizon is not None:
        led = led[led["_h"] == int(horizon)]
    keep = [c for c in ("candidate_id", "stage", "mechanism_tag", "canonical", "_h") if c in led.columns]
    led = led[keep]
    adopted = set(led.loc[led["stage"].astype(str) == "adopted", "candidate_id"].astype(str))
    try:
        from factors.mined_factors import load_adopted
        adopted |= {e["id"] for e in load_adopted()}
    except Exception:
        pass
    out = {}
    for _, r in led[led["stage"].astype(str) == "full"].iterrows():
        cid = str(r["candidate_id"])
        out[cid] = {"adopted": cid in adopted, "horizon": int(r["_h"]),
                    "mechanism_tag": "" if pd.isna(r.get("mechanism_tag")) else str(r.get("mechanism_tag")),
                    "canonical": "" if pd.isna(r.get("canonical")) else str(r.get("canonical"))}
    return out


def recent_notes(runs_dir: Path = RUNS, n: int = RECENT_RUNS, cap: int = NOTES_PER_RUN) -> list[tuple[str, str]]:
    out = []
    for d in reversed(_run_dirs(runs_dir)):
        f = d / "notes.md"
        if f.exists():
            text = f.read_text(encoding="utf-8")
            if len(text) > cap:
                text = text[:cap] + f"\n...（截断，原文 {len(text)} 字）"
            out.append((d.name, text))
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------
def build_memory(ledger_path: Path, runs_dir: Path = RUNS, max_chars: int = MAX_CHARS,
                 horizon: int | None = None) -> str:
    """The memory shown to a run at `horizon`: beliefs and outcomes from
    OTHER horizons are shown too (they are evidence), but labelled, and the
    expression index is horizon-tagged so the agent knows what was tested
    where."""
    beliefs, runs = load_beliefs(runs_dir)
    idx = expression_index(ledger_path)
    outcomes = full_stage_outcomes(ledger_path)
    not_adopted_tags = {(v["mechanism_tag"].lower(), v["horizon"]) for v in outcomes.values()
                        if not v["adopted"] and v["mechanism_tag"]}
    parts = []
    parts.append(f"# 研究记忆（harness 自动生成；{len(runs)} 轮写过 beliefs，{len(idx)} 个表达式已筛选，"
                 f"{len(outcomes)} 个候选进过完整阶段"
                 + (f"；本轮持有期 {horizon} 日" if horizon else "") + "）")
    if horizon:
        parts.append(f"注意持有期：下表带持有期标注。在其他持有期上判 dead 的机制在 {horizon} 日上不一定 dead，"
                     f"但它的证据仍然有用；同一表达式在不同持有期是不同的检验，索引里的重复判定只在同一持有期内生效。")
    parts.append("只含开发段信息、历轮 agent 自己的笔记，以及每个进过完整阶段的候选\"是否被采纳\"这一个比特。"
                 "隐藏段的数字不在这里，也不会在任何地方给你。")

    if outcomes:
        parts.append("\n## 完整阶段结果（预注册的 1 比特反馈：只有采纳与否）")
        parts.append("| 候选 | 机制 | 持有期 | 结果 |")
        parts.append("|---|---|---|---|")
        for cid, v in outcomes.items():
            parts.append(f"| {cid} | {v['mechanism_tag'] or '（无标签）'} | {v['horizon']} | {'已采纳' if v['adopted'] else '未采纳'} |")
        if not_adopted_tags:
            parts.append("未采纳意味着该候选在你看不到的证据段上没有通过规则手册的门槛。"
                         "同一机制的其他写法大概率同样过不去；除非机制本身有实质不同，否则视为 dead。")

    parts.append("\n## 机制状态（各轮 beliefs.json 合并，后写覆盖先写）")
    if beliefs:
        order = {"promising": 0, "weak": 1, "untested": 2, "dead": 3}
        rows = sorted(beliefs.values(), key=lambda b: (order[b["status"]], b["mechanism"]))
        parts.append("| 机制 | 持有期 | 状态 | 完整阶段 | 证据 | 下一步 | 轮次 |")
        parts.append("|---|---|---|---|---|---|---|")
        for b in rows:
            fs = "未采纳" if (b["mechanism"].lower(), b["horizon"]) in not_adopted_tags else ""
            parts.append(f"| {b['mechanism']} | {b['horizon']} | {b['status']} | {fs} | {b['evidence'][:160]} | {b['next'][:160]} | {b['run_id']} |")
    else:
        parts.append("（尚无 beliefs 记录）")

    parts.append("\n## 已筛选表达式索引（重复提交会直接返回旧结果，不再计算）")
    if len(idx):
        parts.append("| canonical | 持有期 | 方向 | dev_ic | dev_t | 过筛 | 冗余于 | 残差t | 轮次 |")
        parts.append("|---|---|---|---|---|---|---|---|---|")
        idx2 = idx.copy()
        idx2["abs_t"] = idx2["dev_t"].abs()
        for _, r in idx2.sort_values("abs_t", ascending=False).iterrows():
            src = str(r.get("source", ""))
            run = src.split(":", 1)[1] if ":" in src else src
            ic = "" if pd.isna(r["dev_ic"]) else f"{r['dev_ic']:+.4f}"
            t = "" if pd.isna(r["dev_t"]) else f"{r['dev_t']:+.2f}"
            sp = "" if pd.isna(r.get("screen_pass")) else ("是" if r["screen_pass"] is True else "否")
            red = "" if pd.isna(r.get("redundant_with")) else str(r.get("redundant_with"))
            rt = "" if pd.isna(r.get("residual_dev_t")) else f"{float(r['residual_dev_t']):+.2f}"
            hz = int(pd.to_numeric(r.get("horizon"), errors="coerce")) if pd.notna(r.get("horizon")) else 20
            parts.append(f"| `{r['canonical']}` | {hz} | {r['expected_direction']} | {ic} | {t} | {sp} | {red} | {rt} | {run} |")
    else:
        parts.append("（尚无筛选记录）")

    notes = recent_notes(runs_dir)
    if notes:
        parts.append(f"\n## 最近 {len(notes)} 轮笔记（原文）")
        for run_id, text in notes:
            parts.append(f"\n### {run_id}\n{text}")

    doc = "\n".join(parts)
    if len(doc) > max_chars:
        doc = doc[:max_chars] + f"\n...（记忆文档截断于 {max_chars} 字；表达式索引和机制表在前面完整保留优先）"
    return doc


# --------------------------------------------------------------------------
# learning audit (human-side): did a run act on the memory it was given?
# --------------------------------------------------------------------------
def _run_proposals(run_dir: Path) -> list[dict]:
    out = []
    for f in sorted(Path(run_dir).glob("proposals*.json")):
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = body.get("proposals", body) if isinstance(body, dict) else body
        out.extend(i for i in items if isinstance(i, dict))
    return out


def audit_run(run_dir: Path, ledger_path: Path, runs_dir: Path = RUNS, horizon: int | None = None) -> dict:
    """Measure whether a run used its memory: proposals aimed at mechanisms
    that EARLIER runs had marked dead, expressions that duplicated the index
    from earlier runs, whether reflection.md and beliefs.json exist, and
    which beliefs changed status this run. Human-side; dev-only inputs."""
    run_dir = Path(run_dir)
    run_name = run_dir.name
    from mining import dsl
    # beliefs as they stood BEFORE this run (every other run's beliefs, in order)
    props = _run_proposals(run_dir)
    run_h = int(horizon or (props[0].get("horizon") if props else 20) or 20)
    before: dict[str, dict] = {}
    for d in _run_dirs(runs_dir):
        if d.name == run_name:
            continue
        f = d / "beliefs.json"
        if not f.exists():
            continue
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
            bh = int(body.get("horizon") or 20)
            for b in body.get("beliefs", []):
                if int(b.get("horizon") or bh) != run_h:
                    continue                  # dead at another horizon is not dead here
                before[str(b.get("mechanism", "")).strip().lower()] = str(b.get("status", "")).lower()
        except Exception:
            pass
    dead_before = {m for m, s in before.items() if s == "dead"}

    idx = expression_index(ledger_path)
    if len(idx):
        from evaluation import rulebook as rb
        idx = idx[rb.horizon_of(idx) == run_h]
    earlier = idx[~idx["source"].astype(str).str.endswith(run_name)] if len(idx) else idx
    earlier_canon = set(earlier["canonical"].astype(str)) if len(earlier) else set()

    hits_dead, dup_exprs, tagged = [], [], 0
    for p in props:
        tag = str(p.get("mechanism_tag", "")).strip().lower()
        if tag:
            tagged += 1
            if tag in dead_before:
                hits_dead.append(p.get("candidate_id"))
        try:
            if dsl.canonical(str(p.get("expression", ""))) in earlier_canon:
                dup_exprs.append(p.get("candidate_id"))
        except Exception:
            pass

    after: dict[str, str] = {}
    bf = run_dir / "beliefs.json"
    if bf.exists():
        try:
            for b in json.loads(bf.read_text(encoding="utf-8")).get("beliefs", []):
                after[str(b.get("mechanism", "")).strip().lower()] = str(b.get("status", "")).lower()
        except Exception:
            pass
    changed = {m: f"{before[m]} -> {s}" for m, s in after.items() if m in before and before[m] != s}
    new_mechs = [m for m in after if m not in before]

    return {
        "run_id": run_name, "horizon": run_h,
        "n_proposals": len(props),
        "n_tagged_with_mechanism": tagged,
        "reflection_written": (run_dir / "reflection.md").exists(),
        "beliefs_written": bf.exists(),
        "dead_mechanisms_known_before": sorted(dead_before),
        "proposals_aimed_at_dead_mechanisms": hits_dead,
        "proposals_duplicating_earlier_expressions": dup_exprs,
        "beliefs_changed_this_run": changed,
        "new_mechanisms_this_run": new_mechs,
        "learned": (len(hits_dead) == 0 and len(dup_exprs) == 0
                    and (run_dir / "reflection.md").exists() and bf.exists()),
    }


if __name__ == "__main__":       # pragma: no cover
    import sys
    from evaluation import rulebook as rb
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(build_memory(rb.MINED_LEDGER))
