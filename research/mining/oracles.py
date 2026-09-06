"""Process-level oracles: runtime self-checks of the mining harness.

The evaluator's own tests pin its code; these checks pin its *behaviour*
on every candidate it screens, whatever produced the feature. They are the
dynamic complement to "leakage cannot be written": if something -- a harness
bug, a mutation, a future operator added carelessly -- lets a feature read
the future, misaligns the label, drops the point-in-time mask, or leaks a
hidden tier into the agent's view, one of these fires and the candidate is
quarantined (never eligible for the full stage) or the harness refuses to
emit its output.

  future_perturbation   metamorphic: recompute the feature on a copy of the
                        panel whose rows after a cutoff are randomly
                        perturbed; every pre-cutoff value must be identical.
                        Catches shift(-k), centred windows, full-period
                        normalisers, whatever the producer is.
  strength              plausibility: |dev t| or |dev IC| beyond what any
                        honest feature has ever shown on this ruler is an
                        audit signal, not a discovery (RULEBOOK, "expected
                        outcome").
  membership            N-version: the dev IC recomputed on an independently
                        derived point-in-time mask must equal the official
                        one; a harness that scores non-member rows disagrees.
  controls              label health: expressions whose honest IC band is
                        known (today's return, log volume, log close) are
                        screened alongside the batch; a misaligned label
                        pushes them far outside the band.
  visibility            canary: nothing outside AGENT_VISIBLE, and no hidden
                        numeric value, may appear in the text handed to the
                        agent.

Thresholds are pre-registered here and in RULEBOOK.md; the mutation-injection
experiment (evaluation/experiments/mutation_injection.py) measures which
injected defects each oracle detects and the false-alarm rate on clean runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STRENGTH_T = 6.0            # honest raw features on this ruler: |t| <= ~3
STRENGTH_IC = 0.08          # honest raw features: |IC| <= ~0.03
CONTROL_BAND = 0.05         # |dev IC| of a control expression beyond this = label alarm
CONTROLS = ("returns", "log(volume)", "log(close)")
PERTURB_SYMBOLS = 60
PERTURB_CUTOFF = 0.7
PERTURB_COLS = ("open", "high", "low", "close", "volume", "marketcap", "turnover", "filing_days")
RESULTS = Path(__file__).resolve().parent.parent / "evaluation" / "results"
PIT_INDEPENDENT = RESULTS / "_mining_pit_independent.parquet"


# --------------------------------------------------------------------------
def future_perturbation(expr: str, panel: pd.DataFrame, compile_fn=None,
                        n_syms: int = PERTURB_SYMBOLS, cutoff_frac: float = PERTURB_CUTOFF,
                        seed: int = 0) -> dict:
    """Perturb every row after a cutoff date (a random symbol subset) and
    recompile; pre-cutoff values must not move. `compile_fn` defaults to the
    DSL compiler *as currently bound* so a patched producer is what gets
    tested."""
    from mining import dsl
    fn = compile_fn or dsl.compile_expression
    rng = np.random.default_rng(seed)
    syms = np.sort(panel["symbol"].unique())
    pick = rng.choice(syms, size=min(n_syms, len(syms)), replace=False)
    sub = panel[panel["symbol"].isin(pick)].copy()
    dates = np.sort(sub["date"].unique())
    cutoff = dates[int(len(dates) * cutoff_frac)]
    fut = (sub["date"] >= cutoff).to_numpy()
    clean = np.asarray(fn(expr, sub), dtype=float)
    mut = sub.copy()
    for c in PERTURB_COLS:
        if c in mut.columns:
            vals = mut[c].to_numpy(dtype=float)
            vals[fut] = vals[fut] * rng.uniform(0.5, 1.5, fut.sum())
            mut[c] = vals
    pert = np.asarray(fn(expr, mut), dtype=float)
    a, b = clean[~fut], pert[~fut]
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    diff[one_nan] = np.inf
    max_diff = float(np.nanmax(diff)) if diff.size else 0.0
    return {"fired": bool(max_diff > 1e-9), "max_abs_diff": max_diff,
            "n_checked": int((~fut).sum()), "n_symbols": int(len(pick))}


def strength(dev_ic: float, dev_t: float) -> dict:
    ic, t = float(dev_ic or 0.0), float(dev_t or 0.0)
    fired = abs(t) > STRENGTH_T or abs(ic) > STRENGTH_IC
    return {"fired": bool(fired), "dev_ic": ic, "dev_t": t,
            "bands": {"t": STRENGTH_T, "ic": STRENGTH_IC}}


def independent_pit_mask(panel: pd.DataFrame, cache: Path = PIT_INDEPENDENT,
                         membership_path: Path | None = None) -> np.ndarray:
    """Point-in-time membership recomputed from the membership file by code
    that does not share the harness's `pit` column; cached per panel shape.
    `membership_path` selects the universe's membership table (S&P 500 by
    default); pass a universe-specific `cache` with it."""
    if cache.exists():
        c = pd.read_parquet(cache)
        if len(c) == len(panel) and (c["symbol"].to_numpy()[:5] == panel["symbol"].to_numpy()[:5]).all():
            return c["pit"].to_numpy(dtype=bool)
    from evaluation.experiments.pit_universe_test import membership_mask
    m = membership_mask(panel[["date", "symbol"]], membership_path=membership_path).to_numpy(dtype=bool)
    try:
        pd.DataFrame({"symbol": panel["symbol"].to_numpy(), "pit": m}).to_parquet(cache, index=False)
    except Exception:
        pass
    return m


def membership(panel: pd.DataFrame, feat: np.ndarray, label_col: str, dev_selector,
               official_ic: float, ic_fn, min_names: int,
               membership_path: Path | None = None, cache: Path | None = None) -> dict:
    """Recompute the dev IC on the independently derived PIT mask."""
    pit = independent_pit_mask(panel, cache=cache or PIT_INDEPENDENT, membership_path=membership_path)
    sub = pd.DataFrame({"date": panel["date"].to_numpy(), label_col: panel[label_col].to_numpy(),
                        "_f": np.asarray(feat, dtype=float)})[pit]
    sub["date"] = pd.to_datetime(sub["date"])
    if sub["date"].dt.tz is None and panel["date"].dt.tz is not None:
        sub["date"] = sub["date"].dt.tz_localize(panel["date"].dt.tz)
    dev = dev_selector(sub)
    ic = ic_fn(dev, "_f", target_col=label_col, min_names_per_date=min_names)
    indep = float(ic.mean()) if len(ic) else float("nan")
    diff = abs(indep - float(official_ic)) if np.isfinite(indep) and np.isfinite(official_ic) else float("inf")
    return {"fired": bool(diff > 1e-9), "official_ic": float(official_ic), "independent_ic": indep}


def controls(panel: pd.DataFrame, label_col: str, dev_selector, ic_fn, min_names: int,
             compile_fn=None) -> dict:
    """Screen the control expressions; any |dev IC| beyond the band fires."""
    from mining import dsl
    fn = compile_fn or dsl.compile_expression
    pit = panel["pit"].to_numpy(dtype=bool)
    out, fired = {}, False
    for expr in CONTROLS:
        try:
            f = np.asarray(fn(expr, panel), dtype=float)
        except Exception as e:  # noqa: BLE001
            out[expr] = {"error": str(e)[:80]}
            continue
        sub = pd.DataFrame({"date": panel["date"].to_numpy(), label_col: panel[label_col].to_numpy(),
                            "_f": f})[pit]
        sub["date"] = pd.to_datetime(sub["date"])
        if sub["date"].dt.tz is None and panel["date"].dt.tz is not None:
            sub["date"] = sub["date"].dt.tz_localize(panel["date"].dt.tz)
        ic = ic_fn(dev_selector(sub), "_f", target_col=label_col, min_names_per_date=min_names)
        v = float(ic.mean()) if len(ic) else float("nan")
        out[expr] = {"dev_ic": v, "fired": bool(abs(v) > CONTROL_BAND)}
        fired |= out[expr]["fired"]
    return {"fired": fired, "band": CONTROL_BAND, "controls": out}


def visibility(view: dict, row: dict, visible_keys: tuple, hidden_prefixes=("holdout", "virgin", "fresh"),
               hidden_keys=("verdict", "reasons", "family_n", "bh_threshold", "holdout_p_onesided")) -> dict:
    """Nothing hidden may reach the agent: neither a key nor a number."""
    bad_keys = [k for k in view if k not in visible_keys]
    text = json.dumps(view, default=str)
    leaked_values = []
    for k, v in row.items():
        if not (k.startswith(hidden_prefixes) or k in hidden_keys):
            continue
        if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) and v != 0:
            for fmt in (f"{float(v):.4f}", f"{float(v):.6f}", repr(float(v))):
                if fmt in text:
                    leaked_values.append(k)
                    break
        elif isinstance(v, str) and v and k == "verdict" and v in ("PASS", "FAIL") and v in text:
            leaked_values.append(k)          # full-stage verdicts only; screen verdicts are dev-derived
        elif isinstance(v, str) and v and k == "reasons" and str(row.get("stage")) in ("full", "adopted") and v in text:
            leaked_values.append(k)
    fired = bool(bad_keys or leaked_values)
    return {"fired": fired, "bad_keys": bad_keys, "leaked_values": leaked_values}
