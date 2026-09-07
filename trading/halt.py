"""Account-level trading halt (circuit breaker state), persisted on disk.

The 2026-09 review found the daily-loss breaker only flattened positions
inside the (long dead) monitor process and left no state behind: the
next open leg would happily re-risk the account. Now:

  * any entry point that adds risk (orchestrator open leg, monitor's
    limit-to-market conversion) must call is_halted() first;
  * the breaker is evaluated by the orchestrator itself from the account's
    equity vs last_equity, so it does not depend on the monitor running;
  * a halt persists until a human clears it (run_staggered_trading.py
    --clear-halt) -- there is deliberately no automatic reset.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
HALT_PATH = ROOT / "trading_logs" / "halt_state.json"


def _resolve(path: Optional[Path]) -> Path:
    # resolved at CALL time so tests (and ops) can repoint HALT_PATH on the module
    return Path(path) if path is not None else Path(HALT_PATH)


def read(path: Optional[Path] = None) -> dict:
    path = _resolve(path)
    if not Path(path).exists():
        return {"halted": False}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        # an unreadable halt file must fail SAFE: treat as halted
        return {"halted": True, "reason": "halt file unreadable", "triggered_at": None}


def is_halted(path: Optional[Path] = None) -> bool:
    return bool(read(path).get("halted"))


def set_halt(reason: str, equity: Optional[float] = None, extra: Optional[dict] = None,
             path: Optional[Path] = None) -> dict:
    path = _resolve(path)
    state = {"halted": True, "reason": reason, "triggered_at": datetime.now().isoformat(timespec="seconds"),
             "equity_at_trigger": equity, "cleared_at": None, "cleared_by": None, **(extra or {})}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)
    return state


def clear(by: str, path: Optional[Path] = None) -> dict:
    path = _resolve(path)
    state = read(path)
    state.update({"halted": False, "cleared_at": datetime.now().isoformat(timespec="seconds"), "cleared_by": by})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)
    return state


def daily_loss_breached(equity: float, last_equity: float, max_loss_pct: float) -> bool:
    """max_loss_pct as a fraction (0.03 = 3%)."""
    if not last_equity or last_equity <= 0:
        return False
    return (equity - last_equity) / last_equity <= -abs(max_loss_pct)
