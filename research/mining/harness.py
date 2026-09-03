"""Two-stage evaluation harness for agent-mined factors (rulebook track B).

    python mining/harness.py [--horizon 5|20] <command>       one track-B family per horizon (default 20)
    python mining/harness.py baseline                         HUMAN: incumbent walk-forward at --horizon
    python mining/harness.py ops                              operator registry (for the agent)
    python mining/harness.py memory                           cross-run research memory (dev-only)
    python mining/harness.py audit-run RUN_ID                 HUMAN: did the run learn from memory?
    python mining/harness.py screen proposals.json [--out f]  stage 1: seen_dev standalone rank IC
    python mining/harness.py full proposals.json --id CID     stage 2: purged walk-forward, hidden tiers
    python mining/harness.py show CID                         HUMAN: reveal the hidden record
    python mining/harness.py adopt CID --removal-trigger T    HUMAN: register a PASS into production

What the agent sees. `screen` and `full` print JSON containing only the
seen_dev numbers (AGENT_VISIBLE). Every other tier -- virgin_early,
holdout, fresh -- plus the BH verdict is written to
evaluation/results/mined_candidates.csv and never printed by those
commands. `show` is the human's reveal.

Sandbox. Splits (SEGMENTS), the label (20-session log return), the
universe (survivorship-complete panel, point-in-time S&P membership), the
evaluator (per-date rank IC, Newey-West t with 19 lags, purged
walk-forward) and the gate (evaluation/rulebook.py) are fixed here and in
the modules this file imports. The agent's only input is a proposal file.

Screen rule (pre-registered, RULEBOOK.md): a candidate is eligible for the
full stage when its signed seen_dev t >= 2.0 in the declared direction and
its dev coverage >= 0.90. Selection on seen_dev is allowed; it is the only
segment tuning may touch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from mining import dsl                                   # noqa: E402
from mining.proposal import Proposal, load_proposals     # noqa: E402
from evaluation import rulebook as rb                    # noqa: E402
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402
from evaluation.factor_card import SEGMENTS              # noqa: E402

DATA = RESEARCH / "data"
RESULTS = RESEARCH / "evaluation" / "results"
SURV_PANEL = DATA / "stocks_with_time_windows_surv.parquet"
MEMBERSHIP = DATA / "sp500_membership.parquet"
BASELINE_TAG = "surv"
PRODUCTION_HORIZON = 20
HORIZONS = (5, 20)

HORIZON = 20
NW_LAGS = HORIZON - 1
MIN_NAMES = 40
LABEL = f"future_return_{HORIZON}d"
SCREEN_T = 2.0
SCREEN_COVERAGE = 0.90


def configure(horizon: int) -> None:
    """Select the label horizon for this process (RULEBOOK: track B runs
    one family per horizon; only the production horizon can reach the
    live book). Everything downstream reads these module globals."""
    global HORIZON, NW_LAGS, LABEL
    if int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    HORIZON = int(horizon)
    NW_LAGS = HORIZON - 1
    LABEL = f"future_return_{HORIZON}d"


def screen_cache_path() -> Path:
    # v2: carries the auxiliary fields; the production horizon keeps the original name
    return RESULTS / ("_mining_screen_panel_v2.parquet" if HORIZON == PRODUCTION_HORIZON
                      else f"_mining_screen_panel_v2_h{HORIZON}.parquet")

AGENT_VISIBLE = ("candidate_id", "stage", "expression", "canonical", "proposal_hash",
                 "lookback", "n_nodes", "dev_ic", "dev_t", "dev_n", "coverage",
                 "blend_dev_gain", "screen_pass", "recorded", "error",
                 "duplicate_of", "note",
                 "max_corr", "corr_with", "cluster_id", "cluster_rep", "redundant_with",
                 "cluster_size", "residual_dev_ic", "residual_dev_t", "residual_vs",
                 "oracle_flags", "quarantined")


# --------------------------------------------------------------------------
# panel helpers
# --------------------------------------------------------------------------
def add_label(df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    """Attach the sealed label: per-symbol log(close[t+horizon] / close[t]).
    Rows must be date-ordered within each symbol."""
    h = int(horizon or HORIZON)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    df[f"future_return_{h}d"] = df.groupby("symbol")["close"].transform(
        lambda x: np.log(x.shift(-h) / x))
    return df


def segment(df: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = next((s, e) for n, s, e in SEGMENTS if n == name)
    tz = df["date"].dt.tz
    m = pd.Series(True, index=df.index)
    if start:
        m &= df["date"] >= pd.Timestamp(start).tz_localize(tz)
    if end:
        m &= df["date"] < pd.Timestamp(end).tz_localize(tz)
    return df[m]


def _members() -> set:
    return set(pd.read_parquet(MEMBERSHIP)["symbol"].unique())


def _pit_mask(df: pd.DataFrame) -> pd.Series:
    from evaluation.experiments.pit_universe_test import membership_mask
    return membership_mask(df)


def load_screen_panel(rebuild: bool = False) -> pd.DataFrame:
    """OHLCV + label for every point-in-time S&P member row of the
    survivorship-complete panel, with a `pit` flag. Features are compiled on
    all member rows (trailing windows need the true history); IC is scored
    on `pit` rows only. Cached because the screen is meant to be cheap."""
    cache = screen_cache_path()
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(SURV_PANEL, columns=cols)
    df = df[df["symbol"].isin(_members())].copy()
    df = add_label(df)
    df["pit"] = _pit_mask(df).to_numpy()
    from mining import aux_fields
    df = aux_fields.attach(df)                 # marketcap / turnover / earn_days when sources exist
    df.to_parquet(cache, index=False)
    return df


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
def base_row(p: Proposal, stage: str) -> dict:
    row = {c: np.nan for c in rb.MINED_COLUMNS}
    row.update({"date": date.today().isoformat(), "candidate_id": p.candidate_id,
                "proposal_hash": p.proposal_hash, "source": p.source,
                "expected_direction": p.expected_direction, "stage": stage,
                "expression": p.expression, "reasons": "",
                "mechanism_tag": p.mechanism_tag, "horizon": HORIZON})
    return row


def agent_view(row: dict) -> dict:
    """The only part of a record the mining agent is allowed to see."""
    return {k: _plain(row[k]) for k in AGENT_VISIBLE if k in row and not _isnan(row[k])}


def _isnan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _plain(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------
# stage 1: screen
# --------------------------------------------------------------------------
def screen_one(p: Proposal, panel: pd.DataFrame, ledger_path: Path = rb.MINED_LEDGER,
               record: bool = True) -> dict:
    row = base_row(p, "screen")
    try:
        stats = p.validate()
        if int(p.horizon) != HORIZON:
            raise ValueError(f"proposal horizon {p.horizon} != harness horizon {HORIZON}")
        # Memory: an expression already screened (any run, SAME horizon) is not
        # re-scored and not re-recorded; the agent gets the old dev record back.
        from mining.memory import find_duplicate
        prior = find_duplicate(ledger_path, stats["canonical"], horizon=HORIZON)
        if prior is not None and prior.get("candidate_id") != p.candidate_id:
            # the old dev statistics are reused; the pass rule is re-applied in
            # THIS proposal's declared direction (a sign flip is not a free pass)
            p_t, p_cov = prior.get("dev_t"), prior.get("coverage")
            p_sign = 1 if p.expected_direction == "positive" else -1
            p_pass = (p_t is not None and p_cov is not None
                      and p_sign * float(p_t) >= SCREEN_T and float(p_cov) >= SCREEN_COVERAGE)
            row.update({"dev_ic": prior.get("dev_ic"), "dev_t": p_t,
                        "coverage": p_cov, "lookback": stats["lookback"],
                        "n_nodes": stats["n_nodes"], "canonical": stats["canonical"],
                        "screen_pass": bool(p_pass),
                        "duplicate_of": prior.get("candidate_id"),
                        "note": f"identical expression already screened as {prior.get('candidate_id')} "
                                f"({prior.get('source')}); returning that record, nothing recomputed",
                        "verdict": "duplicate", "recorded": False})
            return row
        feat = dsl.compile_expression(p.expression, panel)
        pit = panel["pit"].to_numpy(dtype=bool)
        sub = pd.DataFrame({"date": panel.loc[pit, "date"].to_numpy(),
                            LABEL: panel.loc[pit, LABEL].to_numpy(),
                            "_f": feat.to_numpy()[pit]})
        sub["date"] = pd.to_datetime(sub["date"])
        if sub["date"].dt.tz is None and panel["date"].dt.tz is not None:
            sub["date"] = sub["date"].dt.tz_localize(panel["date"].dt.tz)
        dev = segment(sub, "seen_dev")
        ic = daily_rank_ic(dev, "_f", target_col=LABEL, min_names_per_date=MIN_NAMES)
        s = summarize_ic(ic, nw_lags=NW_LAGS)
        coverage = float(dev["_f"].notna().mean()) if len(dev) else 0.0
        sign = 1 if p.expected_direction == "positive" else -1
        signed_t = sign * s["t_stat"] if s["n_days"] else -np.inf
        row.update({"dev_ic": s["ic_mean"], "dev_t": s["t_stat"], "dev_n": s["n_days"],
                    "coverage": coverage, "lookback": stats["lookback"],
                    "n_nodes": stats["n_nodes"], "canonical": stats["canonical"]})
        row["screen_pass"] = bool(signed_t >= SCREEN_T and coverage >= SCREEN_COVERAGE)
        row["verdict"] = "screen_pass" if row["screen_pass"] else "screen_fail"
        # process-level oracles (mining/oracles.py): a firing quarantines the candidate
        from mining import oracles
        flags = {}
        fp = oracles.future_perturbation(p.expression, panel)
        if fp["fired"]:
            flags["future_leak"] = fp["max_abs_diff"]
        st = oracles.strength(s["ic_mean"], s["t_stat"])
        if st["fired"]:
            flags["implausible_strength"] = {"dev_ic": st["dev_ic"], "dev_t": st["dev_t"]}
        mb = oracles.membership(panel, feat.to_numpy(), LABEL, lambda d: segment(d, "seen_dev"),
                                s["ic_mean"], daily_rank_ic, MIN_NAMES)
        if mb["fired"]:
            flags["membership_mask"] = {"official": mb["official_ic"], "independent": mb["independent_ic"]}
        row["oracle_flags"] = json.dumps(flags, default=float) if flags else ""
        row["quarantined"] = bool(flags)
        if flags:
            _oracle_log(p.candidate_id, flags)
        if row["screen_pass"] and not flags:
            row["_feature"] = feat.to_numpy()      # for the batch redundancy pass; never persisted
    except Exception as e:  # the agent gets the message, the ledger keeps it too
        row["error"] = f"{type(e).__name__}: {e}"
        row["verdict"] = "error"
        row["screen_pass"] = False
    if record:
        rb.append_mined(row, ledger_path)
        row["recorded"] = True
    return row


def _incumbent_refs_for(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    from mining import redundancy as rd
    ref = rd.load_incumbent_refs(panel)
    return {c: ref[c].to_numpy(dtype=float) for c in ref.columns}


def cluster_passes(rows: list[dict], panel: pd.DataFrame, ledger_path: Path = rb.MINED_LEDGER,
                   record: bool = True) -> None:
    """Screen-rule refinement (RULEBOOK): cluster this batch's passes against
    incumbent features and earlier representatives on the dev segment; one
    representative per cluster may enter the full stage. Updates rows in
    place and in the ledger."""
    from mining import redundancy as rd
    passes = [r for r in rows if r.get("screen_pass") and "_feature" in r and not r.get("duplicate_of")]
    if not passes:
        return
    pit = panel["pit"].to_numpy(dtype=bool)
    dates_all = pd.to_datetime(panel["date"])
    if dates_all.dt.tz is None and panel["date"].dt.tz is not None:
        dates_all = dates_all.dt.tz_localize(panel["date"].dt.tz)
    tmp = pd.DataFrame({"date": dates_all}).reset_index(drop=True)
    dev_mask = pit & np.isin(np.arange(len(panel)), segment(tmp, "seen_dev").index.to_numpy())
    dates = dates_all.to_numpy()[dev_mask]
    label = panel[LABEL].to_numpy(dtype=float)[dev_mask]
    refs = {k: v[dev_mask] for k, v in _incumbent_refs_for(panel).items()}
    batch_ids = {r["candidate_id"] for r in passes}
    # earlier representatives only: this batch's own rows are already in the
    # ledger (screen_one appended them) and must not count as their own priors
    priors = {k: v[dev_mask] for k, v in rd.prior_representatives(ledger_path, panel, horizon=HORIZON).items()
              if k not in batch_ids}
    batch = [{"candidate_id": r["candidate_id"], "dev_t": r["dev_t"], "x": r["_feature"][dev_mask]}
             for r in passes]
    info = rd.assess(batch, dates, label, refs, priors)
    for r in passes:
        fields = info.get(r["candidate_id"], {})
        r.update(fields)
        if record:
            rb.update_mined(r["candidate_id"], fields, ledger_path)


def assert_representative(candidate_id: str, ledger_path: Path = rb.MINED_LEDGER,
                          force: bool = False) -> None:
    """Full stage is for cluster representatives only (screen rule)."""
    led = rb.load_mined_ledger(ledger_path)
    rows = led[(led["candidate_id"] == candidate_id) & (led["stage"].astype(str) == "screen")]
    if rows.empty or "cluster_rep" not in rows.columns:
        return
    if rb.truthy(rows.iloc[-1].get("quarantined")) is True and not force:
        raise SystemExit(f"{candidate_id} is quarantined by a process oracle "
                         f"({rows.iloc[-1].get('oracle_flags')}); use --force only after a human audit")
    rep = rb.truthy(rows.iloc[-1]["cluster_rep"])
    if rep is False and not force:
        raise SystemExit(f"{candidate_id} is not the representative of its cluster "
                         f"(redundant_with={rows.iloc[-1]['redundant_with']}); use --force to override")


ORACLE_LOG = RESEARCH / "mining" / "runs" / "oracle.log"


def _oracle_log(candidate_id: str, flags: dict) -> None:
    try:
        ORACLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ORACLE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": datetime.now().isoformat(), "candidate_id": candidate_id,
                                "flags": flags}, default=float) + "\n")
    except Exception:
        pass


def screen(proposals: list[Proposal], panel: pd.DataFrame,
           ledger_path: Path = rb.MINED_LEDGER, record: bool = True,
           run_controls: bool = True) -> list[dict]:
    from mining import oracles
    rows = [screen_one(p, panel, ledger_path, record) for p in proposals]
    cluster_passes(rows, panel, ledger_path, record)
    for r in rows:
        r.pop("_feature", None)
    views = []
    for r in rows:
        v = agent_view(r)
        vis = oracles.visibility(v, r, AGENT_VISIBLE)
        if vis["fired"]:
            _oracle_log(r.get("candidate_id", "?"), {"visibility": vis})
            raise RuntimeError(f"visibility oracle: hidden information would reach the agent: {vis}")
        views.append(v)
    if run_controls:
        ctl = oracles.controls(panel, LABEL, lambda d: segment(d, "seen_dev"), daily_rank_ic, MIN_NAMES)
        if ctl["fired"]:
            _oracle_log("_controls", ctl)
            views.append({"candidate_id": "_harness_controls", "error":
                          "label oracle fired: a control expression is far outside its honest IC band; "
                          "the harness is under audit and this batch's scores must not be trusted",
                          "controls": {k: round(v.get("dev_ic", float("nan")), 4) for k, v in ctl["controls"].items()}})
    return views


# --------------------------------------------------------------------------
# stage 2: full walk-forward + hidden adjudication
# --------------------------------------------------------------------------
def prior_family(candidate_id: str, ledger_path: Path = rb.MINED_LEDGER) -> list[float]:
    """One-sided holdout p of every earlier full-stage candidate at THIS
    horizon except this id (a re-run replaces, it does not double-count)."""
    led = rb.load_mined_ledger(ledger_path)
    if led.empty:
        return []
    full = led[(led["stage"].astype(str) == "full") & (led["candidate_id"] != candidate_id)
              & (rb.horizon_of(led) == HORIZON)]
    return rb.family_pvalues_of(full)


def raw_feature_on(oos: pd.DataFrame, expression: str) -> pd.Series:
    """The candidate's raw expression, compiled on the screen panel (true
    history) and aligned to an out-of-sample panel by (date, symbol)."""
    panel = load_screen_panel()
    feat = dsl.compile_expression(expression, panel)
    key = pd.DataFrame({"date": panel["date"].to_numpy(), "symbol": panel["symbol"].to_numpy(),
                        "_raw": feat.to_numpy()})
    merged = oos[["date", "symbol"]].merge(key, on=["date", "symbol"], how="left")
    return pd.Series(merged["_raw"].to_numpy(), index=oos.index)


def adjudicate_full(p: Proposal, tag: str, results_dir: Path = RESULTS,
                    family: list[float] | None = None,
                    ledger_path: Path = rb.MINED_LEDGER,
                    raw_feature: np.ndarray | None = None) -> dict:
    """Score the walk-forward panel `tag` against the baseline on every tier
    and apply the track-B gate. Returns the FULL record (hidden fields
    included); callers decide what to show.

    The standalone tiers (and therefore the direction clause and the BH
    test) are computed on the candidate's RAW expression, the object the
    proposal declared a direction for. The trained group score
    `factor_<id>` is a return *prediction* and is positively signed by
    construction, so it cannot be tested against a declared sign; its tiers
    are recorded separately as model_* for information, and the blend gain
    (which is what production would inherit) still comes from the model."""
    base = pd.read_parquet(results_dir / f"oos_predictions_h{HORIZON}_{BASELINE_TAG}.parquet")
    var = pd.read_parquet(results_dir / f"oos_predictions_h{HORIZON}_{tag}.parquet")
    fcol = f"factor_{p.candidate_id}"
    if fcol not in var.columns:
        raise RuntimeError(f"{fcol} missing from {tag}: the candidate group was skipped in every fold")
    var = var.assign(_raw=(raw_feature if raw_feature is not None
                           else raw_feature_on(var, p.expression).to_numpy()))

    tiers, ic_by_tier, ic_series = {}, {}, {}
    for name, _, _ in SEGMENTS:
        b = summarize_ic(daily_rank_ic(segment(base, name), "pred", target_col=LABEL,
                                       min_names_per_date=MIN_NAMES), nw_lags=NW_LAGS)
        v = summarize_ic(daily_rank_ic(segment(var, name), "pred", target_col=LABEL,
                                       min_names_per_date=MIN_NAMES), nw_lags=NW_LAGS)
        ic_series[name] = daily_rank_ic(segment(var, name), "_raw", target_col=LABEL,
                                        min_names_per_date=MIN_NAMES)
        f = summarize_ic(ic_series[name], nw_lags=NW_LAGS)
        m = summarize_ic(daily_rank_ic(segment(var, name), fcol, target_col=LABEL,
                                       min_names_per_date=MIN_NAMES), nw_lags=NW_LAGS)
        tiers[name] = {"base": b, "blend": v, "alone": f, "model": m}
        ic_by_tier[name] = {"ic_mean": f["ic_mean"], "t_stat": f["t_stat"], "n_days": f["n_days"]}

    dev = tiers["seen_dev"]
    blend_dev_gain = float(dev["blend"]["ic_mean"] - dev["base"]["ic_mean"])
    # v3: two test statistics -- the IC series pooled over every unseen tier
    # (structural path) and over holdout + fresh (probation path)
    pooled = summarize_ic(pd.concat([ic_series[k] for k in rb.UNSEEN_TIERS if k in ic_series]).sort_index(),
                          nw_lags=NW_LAGS)
    recent = summarize_ic(pd.concat([ic_series[k] for k in rb.RECENT_TIERS if k in ic_series]).sort_index(),
                          nw_lags=NW_LAGS)
    fam = prior_family(p.candidate_id, ledger_path) if family is None else list(family)
    gate = rb.gate_mined(ic_by_tier, p.expected_direction, blend_dev_gain, fam, pooled=pooled, recent=recent)

    dev_rows = segment(var, "seen_dev")
    row = base_row(p, "full")
    stats = dsl.validate(dsl.parse(p.expression))
    row.update({
        "lookback": stats["lookback"], "n_nodes": stats["n_nodes"], "canonical": stats["canonical"],
        "dev_ic": dev["alone"]["ic_mean"], "dev_t": dev["alone"]["t_stat"], "dev_n": dev["alone"]["n_days"],
        "coverage": float(dev_rows["_raw"].notna().mean()) if len(dev_rows) else 0.0,
        "blend_dev_gain": blend_dev_gain,
        "model_dev_t": tiers["seen_dev"]["model"]["t_stat"],
        "model_holdout_t": tiers["holdout"]["model"]["t_stat"],
        "virgin_ic": tiers["virgin_early"]["alone"]["ic_mean"],
        "virgin_t": tiers["virgin_early"]["alone"]["t_stat"],
        "virgin_n": tiers["virgin_early"]["alone"]["n_days"],
        "holdout_ic": tiers["holdout"]["alone"]["ic_mean"],
        "holdout_t": tiers["holdout"]["alone"]["t_stat"],
        "holdout_n": tiers["holdout"]["alone"]["n_days"],
        "holdout_p_onesided": gate["holdout_p_onesided"],
        "fresh_ic": tiers["fresh"]["alone"]["ic_mean"],
        "fresh_t": tiers["fresh"]["alone"]["t_stat"],
        "fresh_n": tiers["fresh"]["alone"]["n_days"],
        "pooled_ic": pooled["ic_mean"], "pooled_t": pooled["t_stat"], "pooled_n": pooled["n_days"],
        "pooled_p_onesided": gate["pooled_p_onesided"],
        "recent_ic": recent["ic_mean"], "recent_t": recent["t_stat"], "recent_n": recent["n_days"],
        "recent_p_onesided": gate["recent_p_onesided"],
        "adoption_tier": gate["adoption_tier"], "rule_version": rb.RULE_VERSION,
        "family_n": gate["family_n"], "bh_threshold": gate["bh_threshold"],
        "verdict": gate["verdict"], "reasons": " | ".join(gate["reasons"]),
        "blend_by_tier": json.dumps({k: {"base": t["base"]["ic_mean"], "blend": t["blend"]["ic_mean"]}
                                     for k, t in tiers.items()}),
    })
    return row


def full_one(p: Proposal, max_folds: int | None = None,
             ledger_path: Path = rb.MINED_LEDGER, force: bool = False) -> dict:
    from evaluation.purged_walk_forward import WalkForwardConfig, run_walk_forward
    if not max_folds:
        assert_representative(p.candidate_id, ledger_path, force=force)
    from features.cross_sectional import add_cross_sectional_zscore
    from factors.factor_definitions import FACTOR_GROUPS

    p.validate()
    df = pd.read_parquet(SURV_PANEL)
    sub = df[df["symbol"].isin(_members())].copy()
    from mining import aux_fields
    sub = aux_fields.attach(sub)
    col = f"mined_{p.candidate_id}"
    sub[col] = dsl.compile_expression(p.expression, sub)     # on the true history
    data = sub[_pit_mask(sub).to_numpy()].copy()              # scored on PIT rows
    data = add_cross_sectional_zscore(data, [col], suffix="_xs", winsorize_pct=0.01)
    groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
    groups[p.candidate_id] = [col, col + "_xs"]

    tag = f"mined_{p.candidate_id}" + ("_smoke" if max_folds else "")
    run_walk_forward(WalkForwardConfig(horizon=HORIZON, min_tail_test=15),
                     max_folds=max_folds, df=data, factor_groups=groups, tag=tag)
    if max_folds:
        return {"candidate_id": p.candidate_id, "stage": "smoke", "recorded": False,
                "note": f"smoke run ({max_folds} folds) -- not adjudicated, not in the family"}
    row = adjudicate_full(p, tag, ledger_path=ledger_path)
    rb.append_mined(row, ledger_path)
    row["recorded"] = True
    return row


# --------------------------------------------------------------------------
# human-only commands
# --------------------------------------------------------------------------
def show(candidate_id: str, ledger_path: Path = rb.MINED_LEDGER) -> pd.DataFrame:
    led = rb.load_mined_ledger(ledger_path)
    return led[led["candidate_id"] == candidate_id]


def adopt(candidate_id: str, removal_trigger: str, ledger_path: Path = rb.MINED_LEDGER) -> dict:
    from factors.mined_factors import register_adopted
    rows = show(candidate_id, ledger_path)
    full = rows[rows["stage"].astype(str) == "full"]
    if full.empty:
        raise SystemExit(f"{candidate_id}: no full-stage record")
    last = full.iloc[-1]
    if str(last["verdict"]) != "PASS":
        raise SystemExit(f"{candidate_id}: latest full verdict is {last['verdict']!r}, not PASS "
                         f"({last['reasons']})")
    if not removal_trigger.strip():
        raise SystemExit("a pre-registered removal trigger is required")
    entry = {"id": candidate_id, "expression": str(last["expression"]),
             "expected_direction": str(last["expected_direction"]),
             "proposal_hash": str(last["proposal_hash"]),
             "adopted_on": date.today().isoformat(),
             "lookback": int(last["lookback"]), "family_n": int(last["family_n"]),
             "holdout_p_onesided": float(last["holdout_p_onesided"]),
             "removal_trigger": removal_trigger, "rule_version": rb.RULE_VERSION,
             "horizon": int(last.get("horizon") or PRODUCTION_HORIZON),
             "adoption_tier": str(last.get("adoption_tier") or "structural")}
    if entry["adoption_tier"] == "probation":
        entry["weight_cap"] = 0.025            # half the 5% static fallback; IC-monitor WARN removes it
        entry["removal_trigger"] = "PROBATION: first IC-monitor WARN removes it | " + removal_trigger
    if entry["horizon"] != PRODUCTION_HORIZON:
        entry["status"] = "shelf"        # validated at a non-production horizon; never compiled for the live book
    register_adopted(entry)
    rec = dict(last)
    rec.update({"date": date.today().isoformat(), "stage": "adopted",
                "reasons": f"removal trigger: {removal_trigger}"})
    rb.append_mined(rec, ledger_path)
    return entry


# --------------------------------------------------------------------------
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=PRODUCTION_HORIZON, choices=HORIZONS,
                    help="label horizon in sessions (one track-B family per horizon)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("ops")
    sp.add_parser("memory")
    sp.add_parser("baseline", help="HUMAN: incumbent walk-forward at --horizon on the surv PIT panel")
    au = sp.add_parser("audit-run"); au.add_argument("run_id")
    s = sp.add_parser("screen"); s.add_argument("proposals"); s.add_argument("--out")
    s.add_argument("--rebuild-cache", action="store_true"); s.add_argument("--no-record", action="store_true")
    f = sp.add_parser("full"); f.add_argument("proposals"); f.add_argument("--id", required=True)
    f.add_argument("--max-folds", type=int, default=None)
    f.add_argument("--force", action="store_true", help="run a non-representative anyway (human)")
    h = sp.add_parser("show"); h.add_argument("candidate_id")
    a = sp.add_parser("adopt"); a.add_argument("candidate_id"); a.add_argument("--removal-trigger", required=True)
    args = ap.parse_args(argv)
    configure(args.horizon)

    if args.cmd == "baseline":
        from evaluation.purged_walk_forward import WalkForwardConfig, run_walk_forward
        df = pd.read_parquet(SURV_PANEL)
        sub_ = df[df["symbol"].isin(_members())].copy()
        data = sub_[_pit_mask(sub_).to_numpy()].copy()
        run_walk_forward(WalkForwardConfig(horizon=HORIZON, min_tail_test=15), df=data, tag=BASELINE_TAG)
        print(f"baseline written: oos_predictions_h{HORIZON}_{BASELINE_TAG}.parquet")
        return
    if args.cmd == "ops":
        print(dsl.describe_ops())
        return
    if args.cmd == "memory":
        from mining.memory import build_memory
        print(build_memory(rb.MINED_LEDGER, horizon=HORIZON))
        return
    if args.cmd == "audit-run":
        from mining.memory import audit_run, RUNS
        print(json.dumps(audit_run(RUNS / args.run_id, rb.MINED_LEDGER, horizon=HORIZON), ensure_ascii=False, indent=1))
        return
    if args.cmd == "screen":
        props = load_proposals(args.proposals)
        panel = load_screen_panel(rebuild=args.rebuild_cache)
        out = screen(props, panel, record=not args.no_record)
        text = json.dumps(out, indent=1)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return
    if args.cmd == "full":
        props = {p.candidate_id: p for p in load_proposals(args.proposals)}
        if args.id not in props:
            raise SystemExit(f"{args.id} not in {args.proposals}")
        t0 = datetime.now()
        row = full_one(props[args.id], max_folds=args.max_folds, force=args.force)
        view = agent_view(row) if row.get("stage") == "full" else row
        view["elapsed_s"] = round((datetime.now() - t0).total_seconds())
        print(json.dumps(view, indent=1))
        return
    if args.cmd == "show":
        pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
        rows = show(args.candidate_id)
        if rows.empty:
            print("no records")
        for _, r in rows.iterrows():
            print(f"--- {r['date']} {r['stage']} ---")
            for k, v in r.items():
                if k not in ("date", "stage") and not _isnan(v):
                    print(f"  {k:<20}{v}")
        return
    if args.cmd == "adopt":
        entry = adopt(args.candidate_id, args.removal_trigger)
        print(json.dumps(entry, indent=1))
        print("\nnext: rebuild features (prepare_prediction_data.py), re-run the surv baseline "
              "(experiments/survivorship_universe.py --reuse-panel), retrain (run_retrain_models.py).")


if __name__ == "__main__":
    main()
