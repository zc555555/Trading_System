"""
Promote the best-horizon factor models to production names.

W2-A (2026-05). After train_multi_horizon_models.py trains 18 ensembles
(6 factors x 3 horizons), this script picks the highest-IC horizon for each
factor and copies that ensemble into the default name expected by the
production signal generator (``ensemble_{factor}.pkl``).

It also rewrites ``artifacts/factor_weights.json`` so that:
  - ``factor_scores`` reflects the IC of the CHOSEN horizon (not the 1d IC)
  - ``effective_weights`` is recomputed using the strategy from config.yaml
  - ``per_factor_horizon`` records which horizon won for each factor (audit)

A backup of the pre-promotion ensembles is kept at:
  artifacts/pre_w2a_backup/ensemble_{factor}.pkl

so you can roll back by re-running this script with --rollback.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from factors.factor_definitions import FACTOR_WEIGHTS as STATIC_WEIGHTS
from factors.factor_weighting import compute_ic_proportional, _read_config_strategy


ARTIFACTS = Path(__file__).parent / "artifacts"
MH_DIR = ARTIFACTS / "multi_horizon"
BACKUP_DIR = ARTIFACTS / "pre_w2a_backup"
METRICS_PATH = MH_DIR / "multi_horizon_metrics.json"
WEIGHTS_PATH = ARTIFACTS / "factor_weights.json"


def backup_current_models(factor_names):
    """One-shot backup of ensemble_{factor}.pkl files before we overwrite them."""
    BACKUP_DIR.mkdir(exist_ok=True)
    backed_up = 0
    for f in factor_names:
        src = ARTIFACTS / f"ensemble_{f}.pkl"
        dst = BACKUP_DIR / f"ensemble_{f}.pkl"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            backed_up += 1
    if backed_up:
        print(f"[backup] Saved {backed_up} pre-W2A models -> {BACKUP_DIR}")


def rollback():
    """Restore the pre-W2A models from backup."""
    if not BACKUP_DIR.exists():
        print(f"[rollback] No backup at {BACKUP_DIR}; nothing to do.")
        return
    restored = 0
    for src in BACKUP_DIR.glob("ensemble_*.pkl"):
        dst = ARTIFACTS / src.name
        shutil.copy2(src, dst)
        restored += 1
    print(f"[rollback] Restored {restored} models from {BACKUP_DIR}")


def promote():
    if not METRICS_PATH.exists():
        raise SystemExit(
            f"Missing {METRICS_PATH}. Run train_multi_horizon_models.py first."
        )

    metrics = json.load(open(METRICS_PATH))
    ic_matrix = metrics["ic_matrix"]
    factor_names = sorted(ic_matrix.keys())

    print("=" * 70)
    print("PROMOTE BEST-HORIZON MODELS TO PRODUCTION")
    print("=" * 70)
    print(f"Source matrix from: {METRICS_PATH.name}")
    print(f"Factors: {factor_names}")

    # Decide best horizon per factor (highest IC).
    # Special rule: if all horizons negative, skip (keep current, factor will get 0 weight).
    decisions = {}
    new_scores = {}
    for fname in factor_names:
        cells = {int(h): ic for h, ic in ic_matrix[fname].items()}
        # pick max IC across positive options; if none positive, pick 1d (will get 0 weight)
        positive = {h: ic for h, ic in cells.items() if ic > 0}
        if positive:
            best_h, best_ic = max(positive.items(), key=lambda kv: kv[1])
        else:
            best_h = 1
            best_ic = cells.get(1, min(cells.values()))
        decisions[fname] = best_h
        new_scores[fname] = float(best_ic)
        marker = "*" if best_h != 1 else " "
        print(f"  {marker} {fname:<12} -> {best_h}d  (IC={best_ic:+.4f})")

    # Backup, then copy chosen ensembles to production names
    backup_current_models(factor_names)

    for fname, h in decisions.items():
        src = MH_DIR / f"ensemble_{h}d_{fname}.pkl"
        dst = ARTIFACTS / f"ensemble_{fname}.pkl"
        if not src.exists():
            print(f"  [WARN] {src.name} missing -- production model for "
                  f"{fname} left untouched")
            continue
        shutil.copy2(src, dst)
        print(f"  [copy] {src.name} -> {dst.name}")

    # Recompute effective weights using config-chosen strategy
    cfg_strategy, cfg_floor = _read_config_strategy()
    strategy = cfg_strategy or "ic_sqrt"  # default for W2-A
    floor = cfg_floor if cfg_floor is not None else 0.0
    power_map = {"ic_squared": 2.0, "ic_sqrt": 0.5, "ic_proportional": 1.0}
    power = power_map.get(strategy, 1.0)
    effective = compute_ic_proportional(new_scores, floor=floor, power=power)
    if effective is None:
        print("[WARN] All chosen ICs <= 0; falling back to static weights.")
        effective = dict(STATIC_WEIGHTS)
        strategy_used = "static_fallback"
    else:
        strategy_used = strategy

    payload = {
        "factor_weights_static": dict(STATIC_WEIGHTS),
        "effective_weights": effective,
        "weighting_strategy": strategy_used,
        "training_date": datetime.now().isoformat(),
        "factor_scores": new_scores,
        "per_factor_horizon": {k: int(v) for k, v in decisions.items()},
        "multi_horizon": True,
        # legacy alias
        "factor_weights": dict(STATIC_WEIGHTS),
    }
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Wrote {WEIGHTS_PATH.name}  (strategy={strategy_used}, "
          f"floor={floor})")

    print(f"\n{'factor':<12}{'horizon':<10}{'IC':>10}{'weight':>10}")
    print("-" * 42)
    for f in sorted(effective.keys(), key=lambda k: -effective[k]):
        h = decisions.get(f, 1)
        ic = new_scores.get(f, 0.0)
        w = effective[f]
        marker = "  " if w > 0 else "x "
        print(f"  {marker}{f:<10}{h}d{'':<7}{ic:>+9.4f}{w*100:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Promote best-horizon models")
    parser.add_argument("--rollback", action="store_true",
                        help="Restore pre-W2A models from backup.")
    args = parser.parse_args()

    if args.rollback:
        rollback()
        return
    promote()


if __name__ == "__main__":
    main()
