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
MEMORY_PAGE = 8000           # chars per `memory --part N` page (subagent tool output is truncated well above this)
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
                 "oracle_flags", "quarantined",
                 "pool_size", "pool_corr_max", "pool_corr_with", "residual_vs_pool_t",
                 "incumbent_corr_max", "incumbent_corr_with")


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
    df = aux_fields.attach(df)                 # EDGAR + GDELT news fields when their sources exist
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
        from mining.pool import POOL_SCREEN_T, POOL_COVERAGE
        if not flags and (row["screen_pass"] or (signed_t >= POOL_SCREEN_T and coverage >= POOL_COVERAGE)):
            row["_feature"] = feat.to_numpy()      # batch redundancy (passes) / pool fields; never persisted
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


def dev_rows(panel: pd.DataFrame):
    """(mask, dates, label) of the point-in-time seen_dev rows of the panel."""
    pit = panel["pit"].to_numpy(dtype=bool)
    dates_all = pd.to_datetime(panel["date"])
    if dates_all.dt.tz is None and panel["date"].dt.tz is not None:
        dates_all = dates_all.dt.tz_localize(panel["date"].dt.tz)
    tmp = pd.DataFrame({"date": dates_all}).reset_index(drop=True)
    mask = pit & np.isin(np.arange(len(panel)), segment(tmp, "seen_dev").index.to_numpy())
    return mask, dates_all.to_numpy()[mask], panel[LABEL].to_numpy(dtype=float)[mask]


def pool_fields(rows: list[dict], panel: pd.DataFrame) -> None:
    """Track P: how much each pass adds to the current pool (dev segment)."""
    from mining import pool as pl
    pool = pl.load_pool(HORIZON)
    cands = [r for r in rows if "_feature" in r and not r.get("duplicate_of")]     # v3.2: pool bar, not the screen bar
    if not cands:
        return
    mask, dates, label = dev_rows(panel)
    feats = pl.member_features(panel, pool["members"], HORIZON)
    feats_dev = feats[mask].reset_index(drop=True)
    comp = pl.composite(feats_dev)
    refs = {k: v[mask] for k, v in _incumbent_refs_for(panel).items()}
    for r in cands:
        sign = 1.0 if r.get("expected_direction") == "positive" else -1.0
        r.update(pl.assess_against_pool(r["_feature"][mask], sign, dates, label, feats_dev, comp, NW_LAGS,
                                        incumbents=refs))
    for r in rows:
        r.setdefault("pool_size", len(pool["members"]))


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
    pool_fields(rows, panel)
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
    def _pool(keys):
        parts = [ic_series[k] for k in keys if k in ic_series and len(ic_series[k])]
        return summarize_ic(pd.concat(parts).sort_index() if parts else pd.Series(dtype=float), nw_lags=NW_LAGS)
    pooled = _pool(rb.UNSEEN_TIERS)
    recent = _pool(rb.RECENT_TIERS)
    fam = prior_family(p.candidate_id, ledger_path) if family is None else list(family)
    gate = rb.gate_mined(ic_by_tier, p.expected_direction, blend_dev_gain, fam, pooled=pooled, recent=recent)

    dev_rows_ = segment(var, "seen_dev")
    row = base_row(p, "full")
    if p.expression.startswith("pool:"):
        stats = {"lookback": np.nan, "n_nodes": np.nan, "canonical": p.expression}
    else:
        stats = dsl.validate(dsl.parse(p.expression))
    row.update({
        "lookback": stats["lookback"], "n_nodes": stats["n_nodes"], "canonical": stats["canonical"],
        "dev_ic": dev["alone"]["ic_mean"], "dev_t": dev["alone"]["t_stat"], "dev_n": dev["alone"]["n_days"],
        "coverage": float(dev_rows_["_raw"].notna().mean()) if len(dev_rows_) else 0.0,
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


def walk_forward_for(p: Proposal, max_folds: int | None = None,
                     feature_frame: pd.DataFrame | None = None) -> str:
    """Purged walk-forward of the incumbent groups plus the candidate's own
    group on the surv PIT panel; writes oos_predictions_h<H>_<tag>.parquet
    and returns the tag."""
    from evaluation.purged_walk_forward import WalkForwardConfig, run_walk_forward
    from features.cross_sectional import add_cross_sectional_zscore
    from factors.factor_definitions import FACTOR_GROUPS
    df = pd.read_parquet(SURV_PANEL)
    sub = df[df["symbol"].isin(_members())].copy()
    from mining import aux_fields
    sub = aux_fields.attach(sub)
    col = f"mined_{p.candidate_id}"
    if feature_frame is not None:                             # Track P composite, precomputed on the screen panel
        key = pd.DataFrame({"date": pd.to_datetime(sub["date"]).dt.tz_localize(None).to_numpy(),
                            "symbol": sub["symbol"].to_numpy()}, index=sub.index)
        ff = feature_frame.copy()
        ff["date"] = pd.to_datetime(ff["date"]).dt.tz_localize(None) if pd.to_datetime(ff["date"]).dt.tz is not None \
            else pd.to_datetime(ff["date"])
        merged = key.merge(ff.rename(columns={"value": col}), on=["date", "symbol"], how="left")
        sub[col] = merged[col].to_numpy(dtype=float)
    else:
        sub[col] = dsl.compile_expression(p.expression, sub)     # on the true history
    data = sub[_pit_mask(sub).to_numpy()].copy()              # scored on PIT rows
    data = add_cross_sectional_zscore(data, [col], suffix="_xs", winsorize_pct=0.01)
    groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
    groups[p.candidate_id] = [col, col + "_xs"]
    tag = f"mined_{p.candidate_id}" + ("_smoke" if max_folds else "")
    run_walk_forward(WalkForwardConfig(horizon=HORIZON, min_tail_test=15),
                     max_folds=max_folds, df=data, factor_groups=groups, tag=tag)
    return tag


def full_one(p: Proposal, max_folds: int | None = None,
             ledger_path: Path = rb.MINED_LEDGER, force: bool = False) -> dict:
    if not max_folds:
        assert_representative(p.candidate_id, ledger_path, force=force)
    p.validate()
    tag = walk_forward_for(p, max_folds=max_folds)
    if max_folds:
        return {"candidate_id": p.candidate_id, "stage": "smoke", "recorded": False,
                "note": f"smoke run ({max_folds} folds) -- not adjudicated, not in the family"}
    row = adjudicate_full(p, tag, ledger_path=ledger_path)
    rb.append_mined(row, ledger_path)
    row["recorded"] = True
    return row


# --------------------------------------------------------------------------
# Track P: pool admission and release (RULEBOOK v3.1)
# --------------------------------------------------------------------------
def pool_admit_rows(cands: list[dict], panel: pd.DataFrame, source: str) -> list[dict]:
    """Sequential admission (strongest dev t first): each candidate is scored
    against the pool AS IT STANDS after the previous admissions. `cands`
    rows need candidate_id, expression, expected_direction, dev_t, dev_ic,
    canonical and the screen flags (screen_pass, quarantined, duplicate_of,
    redundant_with). Returns the admitted members."""
    from mining import pool as pl
    mask, dates, label = dev_rows(panel)
    refs = {k: v[mask] for k, v in _incumbent_refs_for(panel).items()}
    pool = pl.load_pool(HORIZON)
    feats = pl.member_features(panel, pool["members"], HORIZON)          # full-panel signed ranks (cached)
    feats_dev = feats[mask].reset_index(drop=True)
    comp = pl.composite(feats_dev)
    have = {m["hash"] for m in pool["members"]}
    all_dates = pd.to_datetime(panel["date"]).to_numpy()
    admitted, new_cols = [], {}
    for r in sorted(cands, key=lambda z: -abs(float(z.get("dev_t") or 0.0))):
        h = dsl.expression_hash(r["expression"])
        if h in have:
            continue
        try:
            feat = dsl.compile_expression(r["expression"], panel).to_numpy(dtype=float)
        except dsl.DSLError as e:
            print(f"  {r['candidate_id']:<28} -> error {e}")
            continue
        sign = 1.0 if r.get("expected_direction") == "positive" else -1.0
        r = dict(r)
        r.update(pl.assess_against_pool(feat[mask], sign, dates, label, feats_dev, comp, NW_LAGS, incumbents=refs))
        ok, why = pl.admissible(r)
        print(f"  {r['candidate_id']:<28} dev_t {float(r.get('dev_t') or 0):+.2f} corr {r.get('pool_corr_max')} "
              f"resid {r.get('residual_vs_pool_t')} -> {why}", flush=True)
        if ok:
            new = pl.admit(HORIZON, [r], source)
            if new:                                    # extend the in-memory pool; the cache is written once at the end
                admitted += new
                have.add(h)
                col = pl.signed_rank(all_dates, feat, sign)
                new_cols[h] = col
                feats[h] = col
                feats_dev = feats[mask].reset_index(drop=True)
                comp = pl.composite(feats_dev)
    if new_cols:
        pl.POOL_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(pl.features_path(HORIZON), index=False)
    return admitted


def pool_candidates_from_run(run_dir: Path) -> list[dict]:
    """Screen views of a finished run joined with its proposals (direction)."""
    props = {}
    for f in sorted(run_dir.glob("proposals_*.json")):
        for p in json.load(open(f, encoding="utf-8")).get("proposals", []):
            props[p["candidate_id"]] = p
    out = []
    for f in sorted(run_dir.glob("screen_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            pr = props.get(r.get("candidate_id"))
            if pr and not r.get("error") and not r.get("duplicate_of"):
                out.append({**r, "expected_direction": pr["expected_direction"], "expression": pr["expression"]})
    return out


def pool_candidates_from_ledger(ledger_path: Path = rb.MINED_LEDGER) -> list[dict]:
    """Every screen pass of this horizon recorded so far (seeding)."""
    led = rb.load_mined_ledger(ledger_path)
    scr = led[(led["stage"].astype(str) == "screen") & (rb.horizon_of(led) == HORIZON)]
    scr = scr[scr["verdict"].astype(str).isin(["screen_pass", "screen_fail"])]
    scr = scr.sort_values("date").drop_duplicates("candidate_id", keep="last")
    return [{k: (None if _isnan(v) else v) for k, v in r.items()} for _, r in scr.iterrows()]


def pool_release(max_folds: int | None = None, ledger_path: Path = rb.MINED_LEDGER) -> dict:
    """Score the composite as one full-stage candidate (pool_h<H>_r<k>)."""
    from mining import pool as pl
    pool = pl.load_pool(HORIZON)
    n = len(pool["members"])
    if n == 0:
        raise SystemExit("empty pool")
    k = len(pool["releases"]) + 1
    cid = f"pool_h{HORIZON}_r{k}"
    panel = load_screen_panel()
    comp = pl.pool_signal(panel, HORIZON, pool)
    ff = pd.DataFrame({"date": pd.to_datetime(panel["date"]).dt.tz_localize(None).to_numpy(),
                       "symbol": panel["symbol"].to_numpy(), "value": comp})
    p = Proposal(candidate_id=cid, expression=f"pool:h{HORIZON}:{n} members",
                 expected_direction="positive",
                 hypothesis="the equal-weight signed-rank composite of the pool predicts the label out of sample",
                 mechanism="breadth: many weak, low-correlation signals combined (RULEBOOK Track P)",
                 refutation_conditions=["release fails the v3 gate"], mechanism_tag="pool",
                 source=f"pool-release:{HORIZON}", horizon=HORIZON)
    tag = walk_forward_for(p, max_folds=max_folds, feature_frame=ff)
    if max_folds:
        return {"candidate_id": cid, "stage": "smoke", "n_members": n}
    var = pd.read_parquet(RESULTS / f"oos_predictions_h{HORIZON}_{tag}.parquet")
    raw = var[["date", "symbol"]].copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None) if pd.to_datetime(raw["date"]).dt.tz is not None \
        else pd.to_datetime(raw["date"])
    raw = raw.merge(ff, on=["date", "symbol"], how="left")
    row = adjudicate_full(p, tag, ledger_path=ledger_path, raw_feature=raw["value"].to_numpy(dtype=float))
    row["n_members"] = n
    rb.append_mined(row, ledger_path)
    pool["releases"].append({"release_id": cid, "n_members": n, "date": datetime.now().strftime("%Y-%m-%d"),
                             "verdict": row["verdict"], "members": [m["hash"] for m in pool["members"]]})
    pl.save_pool(pool)
    return row


# --------------------------------------------------------------------------
# segment-rotation review (RULEBOOK v3, REVIEW_SCHEDULE)
# --------------------------------------------------------------------------

REVIEW_TIER_COLS = ("virgin", "holdout", "fresh")


def _proposal_from_row(r: pd.Series) -> Proposal:
    return Proposal(candidate_id=str(r["candidate_id"]), expression=str(r["expression"]),
                    expected_direction=str(r["expected_direction"]),
                    hypothesis="segment-rotation review of the recorded expression (see its run notes)",
                    mechanism="segment-rotation review of the recorded expression (see its run notes)",
                    refutation_conditions=["see the original proposal"],
                    mechanism_tag=str(r.get("mechanism_tag") or ""), source=f"review:{r.get('source', '')}",
                    horizon=HORIZON)


def _gate_from_row(row: dict, family: list[float]) -> dict:
    ic_by_tier = {f"{k}_early" if k == "virgin" else k: {"ic_mean": row[f"{k}_ic"], "t_stat": row[f"{k}_t"],
                                                          "n_days": row[f"{k}_n"]} for k in REVIEW_TIER_COLS}
    ic_by_tier["seen_dev"] = {"ic_mean": row["dev_ic"], "t_stat": row["dev_t"], "n_days": row["dev_n"]}
    pooled = {"ic_mean": row["pooled_ic"], "t_stat": row["pooled_t"], "n_days": row["pooled_n"]}
    recent = {"ic_mean": row["recent_ic"], "t_stat": row["recent_t"], "n_days": row["recent_n"]}
    return rb.gate_mined(ic_by_tier, row["expected_direction"], row["blend_dev_gain"], family,
                         pooled=pooled, recent=recent)


def review(review_date: str, rerun: bool = False, ledger_path: Path = rb.MINED_LEDGER,
           results_dir: Path = RESULTS, adjudicate=None) -> dict:
    """Re-adjudicate every track-B full-stage candidate of this horizon under
    the segments of `review_date` (rulebook.segments_for): holdout = the
    twelve months before it, fresh = after it, seen_dev grows to it.

    The rotation is applied at runtime only; rulebook.ACTIVE_REVIEW (the
    committed switch) is not touched, and nothing is written to the ledger.
    The whole set is one BH family (each candidate's pooled and recent p,
    every other candidate's p-values as its family). With rerun=True the
    walk-forward of every candidate is recomputed first (the baseline must
    have been re-run with `baseline` beforehand); otherwise the recorded
    out-of-sample predictions are re-scored on the new segments, which is
    exact for the raw-feature tiers and only truncates the model blend
    where the old predictions end. Writes review_<date>_h<H>.{json,md}."""
    global SEGMENTS
    previous = SEGMENTS
    SEGMENTS = rb.segments_for(review_date)
    adjudicate = adjudicate or adjudicate_full
    try:
        led = rb.load_mined_ledger(ledger_path)
        full = led[(led["stage"].astype(str) == "full") & (rb.horizon_of(led) == HORIZON)].copy()
        latest = full.sort_values("date").groupby("candidate_id").tail(1)
        rows = []
        for _, r in latest.iterrows():
            p = _proposal_from_row(r)
            tag = walk_forward_for(p) if rerun else f"mined_{p.candidate_id}"
            row = adjudicate(p, tag, results_dir=results_dir, family=[], ledger_path=ledger_path)
            row["previous_verdict"] = str(r.get("verdict"))
            row["previous_date"] = str(r.get("date"))
            rows.append(row)
        fam_all = rb.family_pvalues_of(pd.DataFrame(rows)) if rows else []
        for row in rows:                       # pass 2: BH over the whole review family
            own = [x for x in (row.get("pooled_p_onesided"), row.get("recent_p_onesided")) if x is not None]
            others = list(fam_all)
            for x in own:
                if x in others:
                    others.remove(x)
            g = _gate_from_row(row, others)
            row.update({"verdict": g["verdict"], "reasons": " | ".join(g["reasons"]), "family_n": g["family_n"],
                        "bh_threshold": g["bh_threshold"], "adoption_tier": g["adoption_tier"]})
        out = {"review_date": review_date, "horizon": HORIZON, "rule_version": rb.RULE_VERSION,
               "segments": [list(x) for x in SEGMENTS], "rerun": rerun, "n_candidates": len(rows),
               "passes": [r["candidate_id"] for r in rows if r["verdict"] == "PASS"], "rows": rows}
        stem = results_dir / f"review_{review_date}_h{HORIZON}"
        stem.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        stem.with_suffix(".md").write_text(_review_markdown(out), encoding="utf-8")
        return out
    finally:
        SEGMENTS = previous


def _review_markdown(out: dict) -> str:
    seg = "; ".join(f"{n}: {s or '...'} .. {e or '...'}" for n, s, e in out["segments"])
    lines = [f"# Segment-rotation review {out['review_date']} (h={out['horizon']}, {out['rule_version']})", "",
             f"Segments: {seg}", f"Re-run walk-forward: {out['rerun']}", "",
             "| candidate | dir | prev | now | tier | dev t | virgin t | holdout t (n) | fresh t (n) | pooled t | recent t | blend gain | reasons |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append("| {cid} | {d} | {pv} | **{v}** | {tier} | {dt:+.2f} | {vt:+.2f} | {ht:+.2f} ({hn:.0f}) | {ft:+.2f} ({fn:.0f}) | {pt:+.2f} | {rt:+.2f} | {bg:+.4f} | {rs} |".format(
            cid=r["candidate_id"], d=r["expected_direction"][:3], pv=r["previous_verdict"], v=r["verdict"],
            tier=r.get("adoption_tier") or "-", dt=r["dev_t"], vt=r["virgin_t"], ht=r["holdout_t"], hn=r["holdout_n"],
            ft=r["fresh_t"], fn=r["fresh_n"], pt=r["pooled_t"], rt=r["recent_t"], bg=r["blend_dev_gain"], rs=r["reasons"]))
    lines += ["", f"Passes: {', '.join(out['passes']) or 'none'}", "",
              "A PASS here is not an adoption: switch rulebook.ACTIVE_REVIEW to this date, commit, re-run the",
              "candidate with `full --force` (ledger row under the rotated segments), then `adopt`."]
    return "\n".join(lines) + "\n"


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
    mem = sp.add_parser("memory")
    mem.add_argument("--part", type=int, default=0, help="page N of the memory document (MEMORY_PAGE chars per page); 0 = whole")
    sp.add_parser("baseline", help="HUMAN: incumbent walk-forward at --horizon on the surv PIT panel")
    au = sp.add_parser("audit-run"); au.add_argument("run_id")
    s = sp.add_parser("screen"); s.add_argument("proposals"); s.add_argument("--out")
    s.add_argument("--rebuild-cache", action="store_true"); s.add_argument("--no-record", action="store_true")
    f = sp.add_parser("full"); f.add_argument("proposals"); f.add_argument("--id", required=True)
    f.add_argument("--max-folds", type=int, default=None)
    f.add_argument("--force", action="store_true", help="run a non-representative anyway (human)")
    h = sp.add_parser("show"); h.add_argument("candidate_id")
    a = sp.add_parser("adopt"); a.add_argument("candidate_id"); a.add_argument("--removal-trigger", required=True)
    pq = sp.add_parser("pool", help="HUMAN: Track P pool -- status | seed | admit --run RUN_ID | release")
    pq.add_argument("action", choices=["status", "seed", "admit", "release"])
    pq.add_argument("--run", help="run id whose screen passes are offered to the pool (admit)")
    pq.add_argument("--max-folds", type=int, default=None)
    rv = sp.add_parser("review", help="HUMAN: re-adjudicate every full-stage candidate under the segments of --date")
    rv.add_argument("--date", required=True, help="review date, e.g. 2026-11-15 (rulebook.REVIEW_SCHEDULE)")
    rv.add_argument("--rerun", action="store_true", help="recompute each candidate's walk-forward first")
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
        doc = build_memory(rb.MINED_LEDGER, horizon=HORIZON)
        if args.part <= 0:
            print(doc)
            return
        pages, buf = [], ""
        for line in doc.splitlines(keepends=True):
            if len(buf) + len(line) > MEMORY_PAGE and buf:
                pages.append(buf); buf = ""
            buf += line
        if buf:
            pages.append(buf)
        n = len(pages)
        if args.part > n:
            print(f"[记忆 第{args.part}页不存在，共{n}页]")
            return
        tail = "，最后一页" if args.part == n else "，继续 --part " + str(args.part + 1)
        print(f"[记忆 第{args.part}/{n}页{tail}]")
        print(pages[args.part - 1])
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
    if args.cmd == "pool":
        from mining import pool as pl
        if args.action == "status":
            print(json.dumps(pl.status(HORIZON), ensure_ascii=False, indent=1, default=str))
            return
        if args.action == "seed":
            panel = load_screen_panel()
            new = pool_admit_rows(pool_candidates_from_ledger(), panel, source="seed:ledger")
            print(f"admitted {len(new)}; pool now {len(pl.load_pool(HORIZON)['members'])} members")
            return
        if args.action == "admit":
            from mining.memory import RUNS
            panel = load_screen_panel()
            new = pool_admit_rows(pool_candidates_from_run(RUNS / args.run), panel, source=f"run:{args.run}")
            print(f"admitted {len(new)}; pool now {len(pl.load_pool(HORIZON)['members'])} members")
            return
        if args.action == "release":
            t0 = datetime.now()
            row = pool_release(max_folds=args.max_folds)
            row = {k: (None if _isnan(v) else v) for k, v in row.items() if k != "blend_by_tier"}
            row["elapsed_s"] = round((datetime.now() - t0).total_seconds())
            print(json.dumps(row, ensure_ascii=False, indent=1, default=str))
            return
    if args.cmd == "review":
        t0 = datetime.now()
        out = review(args.date, rerun=args.rerun)
        print((RESULTS / f"review_{args.date}_h{HORIZON}.md").read_text(encoding="utf-8"))
        print(f"elapsed: {round((datetime.now() - t0).total_seconds())}s")
        return
    if args.cmd == "adopt":
        entry = adopt(args.candidate_id, args.removal_trigger)
        print(json.dumps(entry, indent=1))
        print("\nnext: rebuild features (prepare_prediction_data.py), re-run the surv baseline "
              "(experiments/survivorship_universe.py --reuse-panel), retrain (run_retrain_models.py).")


if __name__ == "__main__":
    main()
