"""Best-effort operator notification.

A desktop toast through scripts/notify_failure.ps1 (the mechanism the
nightly .bat already uses for crashes) plus a durable line in
trading_logs/NOTIFICATIONS.log. Never raises: a notification failure
must not change what the trading code does next. The 2026-09 review asked
for a halt to reach a human without them reading the log by chance;
trading/halt.set_halt calls this.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "notify_failure.ps1"
LOG = ROOT / "trading_logs" / "NOTIFICATIONS.log"


def notify(title: str, body: str, timeout_s: float = 20.0) -> bool:
    """Returns True when the desktop toast was dispatched (not whether it
    was seen). The log line is written regardless."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {title}: {body}\n")
    except Exception:                                # noqa: BLE001
        pass
    if sys.platform != "win32" or not SCRIPT.exists():
        return False
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                        str(title)[:120], str(body)[:400]],
                       timeout=timeout_s, capture_output=True, check=False)
        return True
    except Exception:                                # noqa: BLE001
        return False
