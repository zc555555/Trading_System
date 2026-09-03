"""Adoption rulebook v2 -- two tracks, pre-registered 2026-09-02.

The single home for every multiple-testing rule this project applies to a
factor-adoption decision. Before this module the Sidak bar was computed in
two places (factor_card.py counted non-"finding" ledger rows, factor_pipeline
counted every row); both now import from here, so a rule change is one diff.

Why two tracks. The v1 rule (Sidak family-wise 5% on holdout t over every
hypothesis in the ledger) was written for a family of ~20 human-authored,
literature-backed hypotheses. Agent-scale factor mining (hundreds of
candidates) under the same rule pushes the bar past |t| ~ 3.5 and forbids
adoption by arithmetic. Literature-backed and data-mined candidates carry
different priors and get different, but equally pre-declared, rules.

TRACK A -- literature (human-written hypothesis with a published mechanism)
    unchanged from v1:
    * family  = every non-"finding" row of hypothesis_ledger.csv
    * gate    = holdout |t| >= Sidak bar(family, alpha 5%), two-sided
    * veto    = any tier with t <= -2 (fresh only once it has >= 60 sessions)

TRACK B -- mined (agent-generated or otherwise data-mined candidates)
    * the agent sees ONLY the seen_dev score; virgin/holdout/fresh are
      computed by the harness, written to mined_candidates.csv, never
      returned to the agent
    * every candidate declares expected_direction BEFORE evaluation; the
      holdout test is one-sided in that direction; a candidate whose holdout
      sign disagrees with its declaration FAILS (no post-hoc sign flipping)
    * family  = every track-B candidate that reached the full (walk-forward)
      stage, cumulative over the project; screen-stage rejects are logged
      but are not in the holdout family (dev is the only segment selection
      may optimise, and they were never scored on holdout)
    * gate    = Benjamini-Hochberg FDR q = 0.10 over the family's one-sided
      holdout p-values, AND blend dev gain > 0 (with_ic > base_ic on
      seen_dev), AND holdout >= 120 sessions
    * veto    = any tier with signed t <= -2 in the declared direction
      (fresh only once it has >= 60 sessions)
    * a verdict is computed over the family as of adjudication and recorded;
      it is not re-adjudicated when the family grows except at a scheduled
      review (next: 2026-11)

Not allowed under either track: choosing q or alpha after seeing results,
re-screening on holdout, moving a candidate between tracks after its scores
are known, or reporting a track-B result under the track-A bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RESULTS = Path(__file__).resolve().parent / "results"
LEDGER = RESULTS / "hypothesis_ledger.csv"
MINED_LEDGER = RESULTS / "mined_candidates.csv"

RULE_VERSION = ("v2 (two-track, pre-registered 2026-09-02): "
                "A literature = Sidak FWER 5% on two-sided holdout t; "
                "B mined = BH FDR q=0.10 on one-sided holdout p in the "
                "pre-declared direction + dev blend gain > 0")

FWER_ALPHA = 0.05
FDR_Q = 0.10
NEG_T = -2.0
FRESH_MIN_SESSIONS = 60
HOLDOUT_MIN_SESSIONS = 120
DEFAULT_FAMILY_A = 20          # used only if the ledger file is missing

MINED_COLUMNS = [
    "date", "candidate_id", "proposal_hash", "source", "expected_direction",
    "stage", "dev_ic", "dev_t", "blend_dev_gain",
    "virgin_ic", "virgin_t", "virgin_n",
    "holdout_ic", "holdout_t", "holdout_n", "holdout_p_onesided",
    "fresh_ic", "fresh_t", "fresh_n",
    "family_n", "bh_threshold", "verdict", "reasons",
    "expression", "canonical", "lookback", "n_nodes", "coverage", "screen_pass",
    "blend_by_tier", "mechanism_tag",
    "max_corr", "corr_with", "cluster_id", "cluster_rep", "redundant_with",
    "residual_dev_ic", "residual_dev_t", "residual_vs",
    "model_dev_t", "model_holdout_t",
    "oracle_flags", "quarantined", "horizon",
]
PRODUCTION_HORIZON = 20


# --------------------------------------------------------------------------
# pure statistics
# --------------------------------------------------------------------------
def sidak_bar(n_tests: int, alpha: float = FWER_ALPHA) -> float:
    """Two-sided t threshold giving family-wise error `alpha` over n tests."""
    per_test = 1 - (1 - alpha) ** (1 / max(int(n_tests), 1))
    return float(norm.ppf(1 - per_test / 2))


def bh_reject(pvals, q: float = FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg step-up. Returns a boolean mask over `pvals`
    (original order): True where the null is rejected at FDR q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(ranked <= thresholds)[0]
    if below.size == 0:
        return np.zeros(m, dtype=bool)
    cutoff = ranked[below[-1]]
    return p <= cutoff


def bh_threshold(pvals, q: float = FDR_Q) -> float:
    """The largest p-value BH rejects at FDR q (0.0 if none)."""
    p = np.asarray(pvals, dtype=float)
    mask = bh_reject(p, q)
    return float(p[mask].max()) if mask.any() else 0.0


def onesided_p(t_stat: float, expected_direction: str) -> float:
    """One-sided normal p-value for a t statistic in the declared direction."""
    sign = _direction_sign(expected_direction)
    return float(norm.sf(sign * float(t_stat)))


def _direction_sign(expected_direction: str) -> int:
    d = str(expected_direction).strip().lower()
    if d in ("positive", "pos", "+", "higher_signal_predicts_higher_future_return"):
        return 1
    if d in ("negative", "neg", "-", "higher_signal_predicts_lower_future_return"):
        return -1
    raise ValueError(f"expected_direction must be positive/negative, got {expected_direction!r}")


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------
def family_size_literature(path: Path = LEDGER) -> int:
    """Track A family: every ledger row that is a hypothesis, i.e. whose
    verdict does not start with 'finding' (measurement corrections and
    horizon findings are not adoption tests)."""
    if not Path(path).exists():
        return DEFAULT_FAMILY_A
    rows = pd.read_csv(path)
    return int((~rows["verdict"].astype(str).str.startswith("finding")).sum())


BOOL_COLUMNS = ("screen_pass", "cluster_rep", "quarantined")


def truthy(v) -> bool | None:
    """Parse a ledger boolean that may have round-tripped through CSV as
    True/False, 1/0, 1.0/0.0 or their strings. None for missing."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return None if (isinstance(v, float) and np.isnan(v)) else bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return True
    if s in ("false", "0", "0.0", "no"):
        return False
    return None


def load_mined_ledger(path: Path = MINED_LEDGER) -> pd.DataFrame:
    """The mined ledger with its boolean columns normalised to True/False/NaN
    (object dtype) regardless of how a previous writer serialised them."""
    if not Path(path).exists():
        return pd.DataFrame(columns=MINED_COLUMNS)
    led = pd.read_csv(path)
    for c in BOOL_COLUMNS:
        if c in led.columns:
            led[c] = led[c].map(truthy).astype(object).where(led[c].notna(), np.nan)
    return led


def append_mined(row: dict, path: Path = MINED_LEDGER) -> int:
    """Append one record to the mined ledger under MINED_COLUMNS (extra keys
    are dropped, missing ones are NaN). Returns the new row count."""
    path = Path(path)
    led = load_mined_ledger(path)
    rec = {c: row.get(c, np.nan) for c in MINED_COLUMNS}
    new = pd.DataFrame([rec], columns=MINED_COLUMNS).astype(object)
    # object frames: no dtype inference across all-NA columns (and no pandas
    # FutureWarning about it); the CSV is unaffected
    led = new if led.empty else pd.concat([led.astype(object), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(path, index=False)
    return len(led)


def update_mined(candidate_id: str, fields: dict, path: Path = MINED_LEDGER,
                 stage: str = "screen") -> bool:
    """Update the latest `stage` row of `candidate_id` with `fields` (only
    MINED_COLUMNS keys). Returns False if no such row."""
    path = Path(path)
    led = load_mined_ledger(path)
    hits = led.index[(led["candidate_id"] == candidate_id) & (led["stage"].astype(str) == stage)]
    if len(hits) == 0:
        return False
    i = hits[-1]
    for k, v in fields.items():
        if k in MINED_COLUMNS:
            if k not in led.columns:
                led[k] = np.nan
            led[k] = led[k].astype(object)      # a float NaN column cannot hold strings/bools
            led.loc[i, k] = v
    led.to_csv(path, index=False)
    return True


def horizon_of(led: pd.DataFrame) -> pd.Series:
    """Per-row label horizon; rows written before the column existed are
    production-horizon rows."""
    if "horizon" not in led.columns:
        return pd.Series(PRODUCTION_HORIZON, index=led.index)
    return pd.to_numeric(led["horizon"], errors="coerce").fillna(PRODUCTION_HORIZON).astype(int)


def family_holdout_pvalues(path: Path = MINED_LEDGER, horizon: int | None = None) -> list[float]:
    """Track B family: one-sided holdout p of every candidate that reached
    the full stage, cumulative over the project, ONE FAMILY PER HORIZON."""
    led = load_mined_ledger(path)
    if led.empty:
        return []
    full = led[led["stage"].astype(str) == "full"]
    if horizon is not None:
        full = full[horizon_of(full) == int(horizon)]
    return [float(x) for x in full["holdout_p_onesided"].dropna()]


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def gate_literature(ic_by_tier: dict, family_bar: float) -> dict:
    """Track A (unchanged v1 rule): holdout |t| >= family bar, and no tier
    significantly negative (t <= -2). Fresh vetoes only with >= 60 sessions."""
    ho = ic_by_tier.get("holdout", {})
    fr = ic_by_tier.get("fresh", {})
    neg = [k for k, v in ic_by_tier.items()
           if v.get("t_stat", 0) <= NEG_T
           and (k != "fresh" or v.get("n_days", 0) >= FRESH_MIN_SESSIONS)]
    passes = ho.get("t_stat", 0) >= family_bar and not neg
    return {"track": "A", "rule_version": RULE_VERSION, "family_bar": family_bar,
            "holdout_t": ho.get("t_stat"), "negative_tiers": neg,
            "fresh_n_days": fr.get("n_days", 0),
            "verdict": "PASS" if passes else "FAIL"}


def gate_mined(ic_by_tier: dict, expected_direction: str, blend_dev_gain: float,
               family_pvalues: list[float] | None = None, q: float = FDR_Q) -> dict:
    """Track B gate. `ic_by_tier` maps tier -> {ic_mean, t_stat, n_days} for
    the candidate's standalone rank IC; `family_pvalues` are the one-sided
    holdout p-values of every PRIOR full-stage track-B candidate (this
    candidate is appended here)."""
    sign = _direction_sign(expected_direction)
    ho = ic_by_tier.get("holdout", {})
    reasons = []

    ho_n = int(ho.get("n_days", 0))
    if ho_n < HOLDOUT_MIN_SESSIONS:
        reasons.append(f"holdout has {ho_n} sessions < {HOLDOUT_MIN_SESSIONS}")

    ho_ic = float(ho.get("ic_mean", 0.0))
    if ho_ic * sign <= 0:
        reasons.append("holdout sign disagrees with pre-declared direction")

    if not (float(blend_dev_gain) > 0):
        reasons.append("no blend gain on seen_dev")

    neg = [k for k, v in ic_by_tier.items()
           if sign * float(v.get("t_stat", 0)) <= NEG_T
           and (k != "fresh" or v.get("n_days", 0) >= FRESH_MIN_SESSIONS)]
    if neg:
        reasons.append(f"significantly negative tier(s) in declared direction: {neg}")

    p = onesided_p(float(ho.get("t_stat", 0.0)), expected_direction)
    fam = list(family_pvalues or []) + [p]
    mask = bh_reject(fam, q)
    thr = bh_threshold(fam, q)
    if not mask[-1]:
        reasons.append(f"holdout p={p:.4f} not rejected by BH q={q} over family n={len(fam)}")

    return {"track": "B", "rule_version": RULE_VERSION, "q": q,
            "family_n": len(fam), "bh_threshold": thr,
            "holdout_p_onesided": p, "holdout_t": ho.get("t_stat"),
            "negative_tiers": neg, "fresh_n_days": ic_by_tier.get("fresh", {}).get("n_days", 0),
            "verdict": "PASS" if not reasons else "FAIL", "reasons": reasons}
