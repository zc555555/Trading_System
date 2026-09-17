"""Intraday daily-loss guard (2026-09-17): the one thing the dead monitor
did that still matters now that bracket orders are gone.

Runs every few minutes during the US session as a scheduled task
(scripts/register_intraday_guard.ps1). Each run:

  1. exits at once outside 09:30-16:00 ET or on a closed day;
  2. exits at once when the account is already HALTED (never re-flattens);
  3. reads equity vs last_equity; if the daily loss reaches
     config_trading.MAX_DAILY_LOSS_PCT it persists the halt (trading/halt.py,
     which also raises the desktop notification), cancels every open order
     and flattens every position through trading/execution.flatten_all,
     and writes FAILURES.log; positions that did not close are named.

Nothing else: no per-position stops, no re-entries, no registry edits (the
evening orchestrator's reconcile converges the registry to the flattened
book). A broker outage is logged and the run ends -- the guard fails safe
by doing nothing, never by guessing.
"""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading import execution as ex  # noqa: E402
from trading import halt  # noqa: E402
from trading import trading_calendar as cal  # noqa: E402
from trading.broker import BrokerError  # noqa: E402

FAILURES_LOG = ROOT / "trading_logs" / "FAILURES.log"
GUARD_LOG = ROOT / "trading_logs" / "intraday_guard.log"
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)


def _log(msg: str, path: Path = GUARD_LOG) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:                                # noqa: BLE001
        pass


def in_session(now_et: datetime) -> bool:
    return cal.is_trading_day(now_et.date()) and SESSION_OPEN <= now_et.time() < SESSION_CLOSE


def run_once(broker, now_et: datetime, max_loss_pct: float, sleep: Callable[[float], None] = time.sleep,
             halt_path: Optional[Path] = None, failures_log: Optional[Path] = None, guard_log: Optional[Path] = None,
             timeout_s: float = 90.0, poll_s: float = 2.0) -> dict:
    """One guard tick. Returns {"action": "skip_closed" | "skip_halted" |
    "ok" | "broker_error" | "halted", ...}."""
    hp = halt_path or halt.HALT_PATH
    fl = failures_log or FAILURES_LOG
    gl = guard_log or GUARD_LOG
    if not in_session(now_et):
        return {"action": "skip_closed"}
    if halt.is_halted(hp):
        return {"action": "skip_halted"}
    try:
        acct = broker.account()
    except BrokerError as e:
        _log(f"broker unavailable: {e}", gl)
        return {"action": "broker_error", "error": str(e)}
    equity, last = float(acct["equity"]), float(acct.get("last_equity") or 0.0)
    loss = (equity - last) / last if last > 0 else 0.0
    if not halt.daily_loss_breached(equity, last, max_loss_pct):
        return {"action": "ok", "loss_pct": loss * 100}
    reason = f"daily loss {loss * 100:.2f}% breached {max_loss_pct * 100:.1f}% (intraday guard)"
    halt.set_halt(reason, equity=equity, extra={"last_equity": last, "source": "intraday_guard"}, path=hp)
    _log(f"HALT: {reason}", gl)
    rep = ex.flatten_all(broker, sleep=sleep, timeout_s=timeout_s, poll_s=poll_s)
    line = (f"HALT set by intraday guard: {reason}; canceled {rep['canceled']} orders, closed {len(rep['closed'])}, "
            f"unfilled {rep['unfilled']}, errors {rep['errors']}")
    _log(line, gl)
    try:
        fl.parent.mkdir(parents=True, exist_ok=True)
        with open(fl, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
    except Exception:                                # noqa: BLE001
        pass
    return {"action": "halted", "loss_pct": loss * 100, "flatten": rep}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report the loss, never halt or flatten")
    ap.add_argument("--force", action="store_true", help="with --dry-run: report even outside the session")
    args = ap.parse_args(argv)
    import config_trading
    from trading.broker import AlpacaBroker
    now_et = cal.eastern_now()
    if not in_session(now_et) and not (args.dry_run and args.force):
        return 0
    if halt.is_halted() and not args.dry_run:
        return 0
    try:
        broker = AlpacaBroker()
    except Exception as e:                           # noqa: BLE001
        _log(f"cannot construct broker: {e}")
        return 0
    if args.dry_run:
        try:
            a = broker.account()
            loss = (float(a["equity"]) - float(a["last_equity"])) / float(a["last_equity"]) * 100
            print(f"[guard] {now_et:%H:%M} ET equity {a['equity']:,.0f} vs last {a['last_equity']:,.0f}: {loss:+.2f}% "
                  f"(breaker {config_trading.MAX_DAILY_LOSS_PCT * 100:.1f}%) -- dry run")
        except BrokerError as e:
            print(f"[guard] broker unavailable: {e}")
        return 0
    out = run_once(broker, now_et, config_trading.MAX_DAILY_LOSS_PCT)
    if out["action"] == "halted":
        print(f"[guard] HALTED: {out}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
