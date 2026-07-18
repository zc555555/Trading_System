"""Compute per-position stop-loss and take-profit prices.

W2-C (2026-05). Replaces the old uniform-percentage stops with an
ATR-adaptive scheme: stop distance = K * ATR_14 capped by configurable
percentage floor/ceiling. High-vol names get wider stops, low-vol names
get tighter ones.

Used at tranche-open time by run_staggered_trading.py. The resulting
stop_price/take_price are persisted in TranchePosition; the monitor reads
them directly so the math is set once at entry and never drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config_trading
from .atr_loader import get_atr


@dataclass
class RiskLevels:
    stop_price: float
    take_price: float
    atr: Optional[float]
    basis: str               # 'atr' | 'fixed_pct_fallback'
    stop_pct: float          # implied %, for logging
    take_pct: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_risk_levels(symbol: str, entry_price: float, side: str) -> RiskLevels:
    """Return stop_price + take_price for a new position.

    Direction-aware:
      LONG  -> stop below entry, take above entry
      SHORT -> stop above entry, take below entry

    Strategy precedence (per ``config_trading.USE_ATR_STOPS``):
      1. If ATR available and USE_ATR_STOPS=True:
            stop_pct = clamp(STOP_ATR_MULTIPLE * ATR / entry, floor, ceil)
            take_pct = clamp(TAKE_ATR_MULTIPLE * ATR / entry, floor, ceil)
      2. Else fall back to STOP_LOSS_PCT / TAKE_PROFIT_PCT.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")

    atr = get_atr(symbol) if config_trading.USE_ATR_STOPS else None

    if atr is not None and config_trading.USE_ATR_STOPS:
        raw_stop = config_trading.STOP_ATR_MULTIPLE * atr / entry_price
        raw_take = config_trading.TAKE_ATR_MULTIPLE * atr / entry_price
        stop_pct = _clamp(
            raw_stop,
            config_trading.STOP_PCT_FLOOR,
            config_trading.STOP_PCT_CEILING,
        )
        take_pct = _clamp(
            raw_take,
            config_trading.TAKE_PCT_FLOOR,
            config_trading.TAKE_PCT_CEILING,
        )
        basis = "atr"
    else:
        stop_pct = config_trading.STOP_LOSS_PCT
        take_pct = config_trading.TAKE_PROFIT_PCT
        basis = "fixed_pct_fallback"

    if side == "long":
        stop_price = entry_price * (1 - stop_pct)
        take_price = entry_price * (1 + take_pct)
    else:  # short
        stop_price = entry_price * (1 + stop_pct)
        take_price = entry_price * (1 - take_pct)

    return RiskLevels(
        stop_price=stop_price,
        take_price=take_price,
        atr=atr,
        basis=basis,
        stop_pct=stop_pct,
        take_pct=take_pct,
    )


def position_breached(
    side: str,
    current_price: float,
    stop_price: Optional[float],
    take_price: Optional[float],
) -> Optional[str]:
    """Return ``'STOP_LOSS'`` / ``'TAKE_PROFIT'`` / None.

    Direction-aware breach check, used by the monitor against per-position
    levels stored in the tranche registry.
    """
    if stop_price is None and take_price is None:
        return None
    if side == "long":
        if stop_price is not None and current_price <= stop_price:
            return "STOP_LOSS"
        if take_price is not None and current_price >= take_price:
            return "TAKE_PROFIT"
    elif side == "short":
        if stop_price is not None and current_price >= stop_price:
            return "STOP_LOSS"
        if take_price is not None and current_price <= take_price:
            return "TAKE_PROFIT"
    return None
