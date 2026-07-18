"""
Factor weight resolution.

Single source of truth for "given the per-factor validation ICs, how should
we blend the factor predictions?". Replaces direct imports of
``factor_definitions.FACTOR_WEIGHTS`` so the weighting strategy can change
without touching every consumer.

Strategies
----------
static
    Use the hand-tuned weights from ``factor_definitions.FACTOR_WEIGHTS``.
    Legacy behavior. Picked when the user explicitly trusts their own prior.

ic_proportional   (default after W1.5)
    1. Read per-factor validation IC from ``artifacts/factor_weights.json``.
    2. Clip each IC at ``floor`` (default 0.0) -- factors at or below the
       floor get zero weight.
    3. Normalize remaining factors so weights sum to 1.0.
    4. If all factors are clipped to zero, fall back to ``static``.

ic_squared
    Like ``ic_proportional`` but uses ``max(IC, 0) ** 2`` -- amplifies the
    strongest factor, useful when one factor is dominant.

ic_sqrt
    Uses ``max(IC, 0) ** 0.5`` -- DAMPENS the gap between strong and weak
    factors. Added for W2-A where multi-horizon training inflated one factor's
    IC to 0.65 and linear weighting would give it 88% of the portfolio.
    The square root gives a more diversified, robust blend.

Why ic_proportional is the default
----------------------------------
This matches the "alpha factor combination" approach used by AQR, Two Sigma
research papers, and the Alphalens tutorial: weight each signal by how much
predictive power it actually demonstrated out-of-sample. Negative-IC factors
are treated as noise and excluded rather than inverted -- inverting weak
factors usually just trades on whatever survivorship bias caused the negative
IC, not real alpha.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional


_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_DEFAULT_JSON = _ARTIFACTS_DIR / "factor_weights.json"


# ---------------------------------------------------------------------------
# Pure weight computation
# ---------------------------------------------------------------------------
def compute_ic_proportional(scores: Dict[str, float], floor: float = 0.0,
                            power: float = 1.0) -> Optional[Dict[str, float]]:
    """Convert per-factor validation ICs into normalized portfolio weights.

    Args:
        scores: {factor_name: validation_correlation}
        floor:  Minimum IC to receive any weight (default 0.0 -- only
                positive-IC factors are kept).
        power:  Exponent applied to the clipped IC. ``1.0`` -> linear
                proportional; ``2.0`` -> Sharpe-style squared weighting.

    Returns:
        Dict of weights summing to 1.0, or ``None`` if every factor was
        clipped to zero (caller should fall back).
    """
    clipped: Dict[str, float] = {
        k: max(v - floor, 0.0) ** power for k, v in scores.items()
    }
    total = sum(clipped.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in clipped.items()}


# ---------------------------------------------------------------------------
# Resolver -- the function all consumers should call
# ---------------------------------------------------------------------------
def load_effective_factor_weights(
    json_path: Optional[Path] = None,
    strategy: Optional[str] = None,
    floor: Optional[float] = None,
    log: bool = True,
) -> Dict[str, float]:
    """Return the factor weights that should be used at signal time.

    Resolution order:
        1. Explicit ``strategy`` argument (if provided).
        2. ``research/config.yaml`` -> ``factor_weighting.strategy``.
        3. Default: ``"ic_proportional"``.

    Args:
        json_path: Path to factor_weights.json. Defaults to
                   ``research/artifacts/factor_weights.json``.
        strategy:  Override config. One of {"static", "ic_proportional",
                   "ic_squared"}.
        floor:     Override the IC floor (default 0.0 or whatever the config
                   says).
        log:       Print which strategy was applied (good for audit trails).
    """
    # Local imports keep this module importable even when its consumers
    # haven't been migrated yet.
    from factors.factor_definitions import FACTOR_WEIGHTS as STATIC_WEIGHTS

    cfg_strategy, cfg_floor = _read_config_strategy()
    strategy = (strategy or cfg_strategy or "ic_proportional").lower()
    if floor is None:
        floor = cfg_floor if cfg_floor is not None else 0.0

    if strategy == "static":
        if log:
            print(f"[factor_weights] strategy=static -> using hand-tuned "
                  f"weights from factor_definitions.FACTOR_WEIGHTS")
        return dict(STATIC_WEIGHTS)

    path = Path(json_path) if json_path else _DEFAULT_JSON
    if not path.exists():
        warnings.warn(
            f"[factor_weights] strategy={strategy} requested but {path.name} "
            f"not found. Falling back to static weights.",
            stacklevel=2,
        )
        return dict(STATIC_WEIGHTS)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Prefer pre-computed effective weights when the requested strategy matches
    # what was used at write time (avoids recomputing and prevents drift).
    effective = payload.get("effective_weights")
    stored_strategy = payload.get("weighting_strategy")
    if effective and stored_strategy == strategy:
        if log:
            print(f"[factor_weights] strategy={strategy} -> "
                  f"using cached effective_weights from {path.name}")
        _log_weights(effective, payload.get("factor_scores", {}))
        return dict(effective)

    scores = payload.get("factor_scores")
    if not scores:
        warnings.warn(
            f"[factor_weights] No factor_scores in {path.name}; "
            f"falling back to static.", stacklevel=2,
        )
        return dict(STATIC_WEIGHTS)

    power_map = {"ic_squared": 2.0, "ic_sqrt": 0.5, "ic_proportional": 1.0}
    power = power_map.get(strategy, 1.0)
    weights = compute_ic_proportional(scores, floor=floor, power=power)
    if weights is None:
        warnings.warn(
            f"[factor_weights] All factors below floor={floor}; "
            f"falling back to static weights.", stacklevel=2,
        )
        return dict(STATIC_WEIGHTS)

    if log:
        print(f"[factor_weights] strategy={strategy} floor={floor} -> "
              f"recomputed from factor_scores in {path.name}")
        _log_weights(weights, scores)
    return weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_config_strategy() -> tuple[Optional[str], Optional[float]]:
    """Best-effort read of factor_weighting block from config.yaml.

    Returns (strategy, floor). Either may be None if the config block isn't
    present (in which case caller uses defaults).
    """
    try:
        # Import here so this module doesn't depend on config_loader being
        # already importable (avoids a cycle when train scripts re-enter).
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if not config_path.exists():
            return None, None
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        block = (cfg.get("model", {}) or {}).get("factor_weighting", {}) or {}
        strategy = block.get("strategy")
        floor = block.get("floor")
        if floor is not None:
            floor = float(floor)
        return strategy, floor
    except Exception:
        return None, None


def _log_weights(weights: Dict[str, float], scores: Dict[str, float]) -> None:
    """Pretty-print the weight table for the audit log."""
    print(f"  {'factor':<12} {'IC':>8} {'weight':>8}")
    for name in sorted(weights.keys(), key=lambda k: -weights[k]):
        ic = scores.get(name, float("nan"))
        w = weights[name]
        marker = "  " if w > 0 else "x "
        print(f"  {marker}{name:<10} {ic:>+8.4f} {w*100:>7.1f}%")


if __name__ == "__main__":
    import pprint
    print("=== load_effective_factor_weights() smoke test ===")
    w = load_effective_factor_weights()
    print("\nFinal weights:")
    pprint.pprint(w)
