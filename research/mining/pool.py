"""Track P: a pool of many weak, low-correlation mined factors evaluated as
ONE composite signal (RULEBOOK v3.1, "Track P").

Why a pool: single-factor gates on S&P 500 large caps reject everything
(nine rounds, zero adoptions) because every honest single signal has an IC
of 0.01-0.02. The industry answer is breadth: admit many weak signals on
cheap development-stage criteria, combine them, and test only the
composite out of sample. Multiple testing is then paid once per pool
RELEASE (each release is one member of the horizon's BH family), not once
per factor.

Admission (development segment only, RULEBOOK "Track P"):
    * screen pass in the declared direction, coverage >= 0.90, no oracle flag
    * |rank correlation| with every current member < POOL_CORR_MAX (0.7)
      and with every incumbent production feature < 0.7 (the screen's own
      redundancy pass already enforces the incumbent side)
    * residual dev t against the current composite >= POOL_RESIDUAL_T in the
      declared direction (the candidate adds information the pool lacks)

Composite: for each member, the cross-sectional percentile rank of its
expression per date, centred, multiplied by the declared sign; the
composite is the plain mean over members with a value (at least
POOL_MIN_FRACTION of the members must be non-missing). Equal weights are
deliberate: with tens to hundreds of members, learned weights overfit.

Release: the composite is scored like a full-stage candidate (purged
walk-forward as its own factor group, raw-feature tiers, v3 gate) under
the ledger id pool_h<H>_r<k>; a PASS is adopted like any candidate.

Files: mining/pool/pool_h<H>.json (members, releases) and
mining/pool/pool_features_h<H>.parquet (cached member signals on the
screen panel, keyed by canonical expression).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mining import dsl

POOL_DIR = Path(__file__).resolve().parent / "pool"
POOL_CORR_MAX = 0.7
POOL_RESIDUAL_T = 1.5
POOL_MIN_FRACTION = 0.3
RELEASE_CHECKPOINTS = (25, 50, 100, 200, 400)


def pool_path(horizon: int) -> Path:
    return POOL_DIR / f"pool_h{horizon}.json"


def features_path(horizon: int) -> Path:
    return POOL_DIR / f"pool_features_h{horizon}.parquet"


def load_pool(horizon: int) -> dict:
    p = pool_path(horizon)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"horizon": horizon, "members": [], "releases": []}


def save_pool(pool: dict) -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool_path(pool["horizon"]).write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# signals
# ----------------------------------------------------------------------------
def signed_rank(dates: np.ndarray, x: np.ndarray, sign: float) -> np.ndarray:
    """Per-date percentile rank of x, centred on zero, times the declared sign."""
    s = pd.Series(x, dtype=float)
    r = s.groupby(pd.Series(dates)).rank(pct=True)
    return ((r - 0.5) * sign).to_numpy(dtype=float)


def member_features(panel: pd.DataFrame, members: list[dict], horizon: int,
                    use_cache: bool = True) -> pd.DataFrame:
    """Signed ranks of every member on the panel rows (columns = canonical
    hash), computed once and cached; new members are appended."""
    dates = pd.to_datetime(panel["date"]).to_numpy()
    cache = features_path(horizon)
    have = pd.DataFrame(index=panel.index)
    if use_cache and cache.exists():
        old = pd.read_parquet(cache)
        if len(old) == len(panel):
            have = old
    cols = {}
    for m in members:
        key = m["hash"]
        if key in have.columns:
            cols[key] = have[key].to_numpy(dtype=float)
            continue
        feat = dsl.compile_expression(m["expression"], panel).to_numpy(dtype=float)
        cols[key] = signed_rank(dates, feat, 1.0 if m["expected_direction"] == "positive" else -1.0)
    out = pd.DataFrame(cols, index=panel.index)
    if use_cache and len(members):
        POOL_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


def composite(features: pd.DataFrame) -> np.ndarray:
    if features.shape[1] == 0:
        return np.full(len(features), np.nan)
    arr = features.to_numpy(dtype=float)
    n_ok = np.sum(~np.isnan(arr), axis=1)
    need = max(1, int(np.ceil(POOL_MIN_FRACTION * arr.shape[1])))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isnan(arr), np.nan, arr), axis=1)
    return np.where(n_ok >= need, mean, np.nan)


def pool_signal(panel: pd.DataFrame, horizon: int, pool: dict | None = None) -> np.ndarray:
    pool = pool or load_pool(horizon)
    if not pool["members"]:
        return np.full(len(panel), np.nan)
    return composite(member_features(panel, pool["members"], horizon))


# ----------------------------------------------------------------------------
# admission
# ----------------------------------------------------------------------------
def assess_against_pool(cand: np.ndarray, sign: float, dates: np.ndarray, label: np.ndarray,
                        features: pd.DataFrame, comp: np.ndarray, nw_lags: int) -> dict:
    """Dev-segment statistics of one candidate (raw expression values on the
    dev rows) against the current pool: max |corr| with a member, residual
    t against the composite. Returns agent-visible numbers only."""
    from mining.redundancy import mean_rank_corr, residual_rank_ic
    from evaluation.metrics import summarize_ic
    out = {"pool_size": int(features.shape[1]), "pool_corr_max": 0.0, "pool_corr_with": "",
           "residual_vs_pool_t": np.nan}
    if features.shape[1] == 0:
        return out
    best, best_key = 0.0, ""
    for key in features.columns:
        c = mean_rank_corr(dates, cand, features[key].to_numpy(dtype=float))
        if abs(c) > abs(best):
            best, best_key = c, key
    out["pool_corr_max"] = round(float(best), 3)
    out["pool_corr_with"] = best_key
    s = summarize_ic(residual_rank_ic(dates, cand, comp, label), nw_lags=nw_lags)
    t = float(s["t_stat"]) if s["n_days"] and np.isfinite(s["t_stat"]) else 0.0
    out["residual_vs_pool_t"] = round(sign * t, 3)          # signed: positive = adds in the declared direction
    return out


def admissible(row: dict) -> tuple[bool, str]:
    """Admission rule on a screen row that already carries the pool fields."""
    if not row.get("screen_pass"):
        return False, "screen_fail"
    if row.get("quarantined"):
        return False, "quarantined"
    if row.get("duplicate_of"):
        return False, "duplicate"
    rw = str(row.get("redundant_with") or "")
    if rw.startswith("incumbent:"):
        return False, f"redundant with {rw}"
    if abs(float(row.get("pool_corr_max") or 0.0)) >= POOL_CORR_MAX:
        return False, f"corr {row.get('pool_corr_max')} with pool member {row.get('pool_corr_with')}"
    rt = row.get("residual_vs_pool_t")
    if rt is not None and not (isinstance(rt, float) and np.isnan(rt)) and int(row.get("pool_size") or 0) > 0:
        if float(rt) < POOL_RESIDUAL_T:
            return False, f"residual t vs pool {rt} < {POOL_RESIDUAL_T}"
    return True, "admitted"


def admit(horizon: int, rows: list[dict], source: str) -> list[dict]:
    """Add every admissible screen row to the pool (idempotent on hash).
    `rows` must carry expression / expected_direction / canonical and the
    pool fields; returns the newly admitted members."""
    pool = load_pool(horizon)
    have = {m["hash"] for m in pool["members"]}
    new = []
    for r in rows:
        ok, why = admissible(r)
        if not ok:
            continue
        h = dsl.expression_hash(r["expression"]) if hasattr(dsl, "expression_hash") else r["canonical"]
        if h in have:
            continue
        m = {"candidate_id": r["candidate_id"], "expression": r["expression"], "canonical": r.get("canonical", ""),
             "hash": h, "expected_direction": r["expected_direction"], "source": source,
             "admitted": datetime.now().strftime("%Y-%m-%d"),
             "dev_ic": float(r.get("dev_ic") or 0.0), "dev_t": float(r.get("dev_t") or 0.0),
             "residual_vs_pool_t": r.get("residual_vs_pool_t"), "pool_corr_max": r.get("pool_corr_max"),
             "pool_size_at_admission": int(r.get("pool_size") or 0)}
        pool["members"].append(m)
        have.add(h)
        new.append(m)
    if new:
        save_pool(pool)
    return new


def status(horizon: int) -> dict:
    pool = load_pool(horizon)
    n = len(pool["members"])
    nxt = next((c for c in RELEASE_CHECKPOINTS if c > max([r["n_members"] for r in pool["releases"]] or [0])), None)
    return {"horizon": horizon, "n_members": n, "releases": pool["releases"],
            "next_checkpoint": nxt, "due": bool(nxt and n >= nxt),
            "by_source": pd.Series([m["source"] for m in pool["members"]]).value_counts().to_dict() if n else {}}
