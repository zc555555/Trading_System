"""Mutation-injection experiment: does the mining process detect known
methodology defects? (the process-level-oracle question)

    cd research
    python evaluation/experiments/mutation_injection.py            # full grid
    python evaluation/experiments/mutation_injection.py --quick    # 3 probes

Each mutation re-creates a defect class with a known real-world instance
and is injected AROUND the harness (monkeypatching the feature producer,
the panel, the visible-field list, or the memory), never by editing the
harness. A fixed set of probe expressions is screened under the clean
harness and under each mutation; every oracle firing is recorded. The
result is a detection matrix plus the false-alarm count on the clean run.

PRE-REGISTERED EXPECTATIONS (written before the first run):

    mutation              real instance                    expected detector
    ------------------    -----------------------------    -----------------------------
    feature_future_shift  dpo_20 shift(-11), 2026-07       future_perturbation
    centered_window       AQuA App. B full-day normaliser  future_perturbation
    label_misaligned      label built from close[t-1]      controls (label band); maybe strength
    pit_mask_off          membership look-ahead, 2026-08   membership (independent mask)
    holdout_exposed       selection leakage                 visibility canary (harness refuses)
    dedupe_off_flip       direction calibration            learning audit (duplicate list)
    clean                 --                               nothing (false alarms counted)

A mutation "detected" = at least one oracle fires on at least one probe
(for holdout_exposed: the harness raises). Probes that are themselves
quarantined on the clean run count as false alarms.

Outputs: evaluation/results/mutation_injection_report.json and .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RESEARCH))

from mining import dsl, harness, memory, oracles   # noqa: E402
from mining.proposal import Proposal                # noqa: E402
from evaluation.metrics import daily_rank_ic        # noqa: E402

RESULTS = RESEARCH / "evaluation" / "results"
PROBES = [
    ("p_mom5", "rank(ts_mean(returns, 5))", "positive"),
    ("p_high52", "close / ts_max(high, 250)", "positive"),
    ("p_volspike", "div(ts_max(dollar_volume, 60), ts_mean(dollar_volume, 60))", "negative"),
    ("p_size", "log(marketcap)", "positive"),
    ("p_turn", "rank(turnover)", "negative"),
    ("p_vol20", "ts_std(returns, 20)", "negative"),
    ("p_filing", "filing_days", "negative"),
    ("p_corr", "rank(ts_corr(close, volume, 20))", "negative"),
]
QUICK = PROBES[:3]
ORACLES = ("future_leak", "implausible_strength", "membership_mask", "visibility", "controls", "audit_duplicate")


def _proposal(cid, expr, direction):
    return Proposal(candidate_id=cid, expression=expr, expected_direction=direction,
                    hypothesis="Probe expression for the mutation-injection experiment.",
                    mechanism="Not a hypothesis; a fixed probe screened under injected defects.",
                    refutation_conditions=["n/a"], mechanism_tag="probe", source="mutation_injection")


# --------------------------------------------------------------------------
# mutations: each returns (panel, cleanup) and patches the producer / views
# --------------------------------------------------------------------------
class Mutation:
    name = "clean"
    real_instance = "-"

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        return panel

    def cleanup(self) -> None:
        pass


class FeatureFutureShift(Mutation):
    name, real_instance = "feature_future_shift", "dpo_20 shift(-11) (audit 2026-07)"

    def apply(self, panel):
        self._orig = dsl.compile_expression

        def leaky(expr, df, **kw):
            s = self._orig(expr, df, **kw)
            return s.groupby(df["symbol"]).shift(-5)          # tomorrow+4 in today's row
        dsl.compile_expression = leaky
        return panel

    def cleanup(self):
        dsl.compile_expression = self._orig


class CenteredWindow(Mutation):
    name, real_instance = "centered_window", "AQuA Appendix B full-day normaliser"

    def apply(self, panel):
        self._orig = dsl.compile_expression

        def leaky(expr, df, **kw):
            s = self._orig(expr, df, **kw)
            return s.groupby(df["symbol"]).transform(lambda g: g.rolling(5, center=True, min_periods=1).mean())
        dsl.compile_expression = leaky
        return panel

    def cleanup(self):
        dsl.compile_expression = self._orig


class LabelMisaligned(Mutation):
    name, real_instance = "label_misaligned", "label from close[t-1]: today's return inside the label"

    def apply(self, panel):
        p = panel.copy()
        p = p.sort_values(["date", "symbol"]).reset_index(drop=True)
        prev = p.groupby("symbol")["close"].shift(1)
        fwd = p.groupby("symbol")["close"].shift(-harness.HORIZON)
        p[harness.LABEL] = np.log(fwd / prev)
        return p


class PitMaskOff(Mutation):
    name, real_instance = "pit_mask_off", "today's members applied to history (PIT experiment 2026-08)"

    def apply(self, panel):
        p = panel.copy()
        p["pit"] = True
        return p


class HoldoutExposed(Mutation):
    name, real_instance = "holdout_exposed", "selection leakage: a hidden tier reaches the agent"

    def apply(self, panel):
        self._vis = harness.AGENT_VISIBLE
        self._orig = harness.screen_one
        harness.AGENT_VISIBLE = self._vis + ("holdout_ic", "holdout_t")

        def leaky(p, pnl, ledger_path=None, record=True):
            row = self._orig(p, pnl, ledger_path, record)
            try:
                feat = dsl.compile_expression(p.expression, pnl)
                pit = pnl["pit"].to_numpy(dtype=bool)
                sub = pd.DataFrame({"date": pnl.loc[pit, "date"].to_numpy(), harness.LABEL: pnl.loc[pit, harness.LABEL].to_numpy(),
                                    "_f": feat.to_numpy()[pit]})
                sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(pnl["date"].dt.tz) if sub["date"].dt.tz is None else sub["date"]
                ic = daily_rank_ic(harness.segment(sub, "holdout"), "_f", target_col=harness.LABEL, min_names_per_date=harness.MIN_NAMES)
                row["holdout_ic"], row["holdout_t"] = float(ic.mean()), float(ic.mean() / ic.std() * np.sqrt(len(ic)))
            except Exception:
                pass
            return row
        harness.screen_one = leaky
        return panel

    def cleanup(self):
        harness.AGENT_VISIBLE = self._vis
        harness.screen_one = self._orig


class DedupeOffFlip(Mutation):
    name, real_instance = "dedupe_off_flip", "direction calibration after seeing dev (AQuA step we forbid)"

    def apply(self, panel):
        self._orig = memory.find_duplicate
        memory.find_duplicate = lambda ledger_path, canonical: None
        return panel

    def cleanup(self):
        memory.find_duplicate = self._orig


MUTATIONS = [Mutation(), FeatureFutureShift(), CenteredWindow(), LabelMisaligned(), PitMaskOff(),
             HoldoutExposed(), DedupeOffFlip()]


# --------------------------------------------------------------------------
def run_one(mut: Mutation, panel: pd.DataFrame, probes: list, tmp: Path) -> dict:
    ledger = tmp / f"ledger_{mut.name}.csv"
    run_dir = tmp / f"run_{mut.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fired = {o: [] for o in ORACLES}
    detail = {}
    t0 = time.time()
    p_mut = mut.apply(panel)
    try:
        props = [_proposal(c, e, d) for c, e, d in probes]
        if mut.name == "dedupe_off_flip":
            # resubmit every probe with the opposite declared sign under a new id
            props += [_proposal(c + "_flip", e, "negative" if d == "positive" else "positive") for c, e, d in probes]
        # proposals file for the audit oracle
        (run_dir / "proposals_1.json").write_text(json.dumps({"proposals": [p.to_dict() for p in props]}), encoding="utf-8")
        try:
            views = harness.screen(props, p_mut, ledger, record=True)
        except RuntimeError as e:
            if "visibility oracle" in str(e):
                fired["visibility"].append("harness refused to emit")
                views = []
            else:
                raise
        for v in views:
            cid = v.get("candidate_id")
            if cid == "_harness_controls":
                fired["controls"].append(v.get("controls"))
                continue
            flags = json.loads(v["oracle_flags"]) if v.get("oracle_flags") else {}
            for k in ("future_leak", "implausible_strength", "membership_mask"):
                if k in flags:
                    fired[k].append(cid)
            detail[cid] = {"dev_ic": v.get("dev_ic"), "dev_t": v.get("dev_t"), "flags": flags, "error": v.get("error")}
        # post-hoc learning audit: duplicates across the batch (the flipped resubmissions)
        idx = memory.expression_index(ledger)
        dup = idx["canonical"].duplicated().sum() if len(idx) else 0
        canon = [dsl.canonical(p.expression) for p in props]
        resub = len(canon) - len(set(canon))
        if mut.name == "dedupe_off_flip":
            # with dedupe on, the flipped copies would have returned duplicate_of; here they were re-scored
            re_scored = sum(1 for v in views if str(v.get("candidate_id", "")).endswith("_flip") and "duplicate_of" not in v)
            if re_scored:
                fired["audit_duplicate"].append(f"{re_scored} flipped resubmissions re-scored (dedupe off)")
        elif resub:
            fired["audit_duplicate"].append(resub)
        _ = dup
    finally:
        mut.cleanup()
    return {"mutation": mut.name, "real_instance": mut.real_instance, "elapsed_s": round(time.time() - t0),
            "fired": {k: v for k, v in fired.items() if v}, "detected": any(fired[o] for o in ORACLES),
            "detail": detail}


def main(quick: bool) -> None:
    import tempfile
    probes = QUICK if quick else PROBES
    panel = harness.load_screen_panel()
    tmp = Path(tempfile.mkdtemp(prefix="mutation_injection_"))
    print(f"probes: {len(probes)} | mutations: {len(MUTATIONS)} | tmp: {tmp}")
    rows = []
    for mut in MUTATIONS:
        r = run_one(mut, panel, probes, tmp)
        rows.append(r)
        print(f"  {mut.name:<22} detected={r['detected']!s:<5} fired={list(r['fired'].keys())}  {r['elapsed_s']}s")
    clean = rows[0]
    false_alarms = sum(len(v) for v in clean["fired"].values())
    report = {"generated_at": datetime.now().isoformat(), "probes": probes, "oracles": list(ORACLES),
              "false_alarms_on_clean": false_alarms, "runs": rows,
              "summary": {r["mutation"]: {"detected": r["detected"], "by": list(r["fired"].keys())} for r in rows}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mutation_injection_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = ["# Mutation-injection report", f"*{report['generated_at']}; {len(probes)} probes; false alarms on clean run: {false_alarms}*", "",
          "| mutation | real instance | detected | fired oracles |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['mutation']} | {r['real_instance']} | {'yes' if r['detected'] else 'no'} | {', '.join(r['fired'].keys()) or '-'} |")
    (RESULTS / "mutation_injection_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
