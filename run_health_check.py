#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nightly health check (2026-09 review item 7): did the job succeed, what
data did it see, were the expected artifacts produced, is the book what
the broker says it is.

Runs at the end of scripts/run_trading_scheduled.bat with the trading
run's exit code. Writes one JSON (trading_logs/health_latest.json), one
line per run to trading_logs/HEALTH.log, and on any FAIL a FAILURES.log
line plus a desktop notification. The August incidents (a crash that sat
unnoticed for nine days, a membership table that silently degraded) were
all of the "nothing checked the outputs" kind.

Checks
  nightly_rc   the orchestrator's exit code (0 ok, 2 broker unavailable)
  data         the feature panel's last date == the last completed session
  signals      today's signal file exists, carries that data date, is < 30h
               old and records its selection parameters
  models       model_release.verify (manifest hashes, all factors present)
  book         registry vs broker (dry-run reconcile): orphans / phantoms
  halt         the persisted halt state
  failures     FAILURES.log lines written today
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

LOGS = ROOT / "trading_logs"
HEALTH_LOG = LOGS / "HEALTH.log"
HEALTH_JSON = LOGS / "health_latest.json"
FAILURES_LOG = LOGS / "FAILURES.log"
SIGNAL_MAX_AGE_H = 30.0


def check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def run_checks(now_et, nightly_rc=None, data_path: Path | None = None, artifacts: Path | None = None,
               broker=None, registry=None, halt_path: Path | None = None,
               failures_log: Path | None = None) -> list[dict]:
    from trading import trading_calendar as cal
    checks = []
    last_session = cal.last_completed_session(now_et)

    # nightly exit code
    if nightly_rc is None:
        checks.append(check("nightly_rc", "SKIP", "not provided"))
    elif int(nightly_rc) == 0:
        checks.append(check("nightly_rc", "OK", "exit 0"))
    else:
        checks.append(check("nightly_rc", "FAIL", f"exit {nightly_rc}"))

    # data freshness
    data_path = data_path or ROOT / "research" / "data" / "stocks_with_time_windows.parquet"
    try:
        import pyarrow.parquet as pq
        dates = pq.read_table(str(data_path), columns=["date"]).column("date").to_pandas()
        dmax = dates.max()
        dmax_d = dmax.date() if hasattr(dmax, "date") else dmax
        if not cal.is_trading_day(now_et.date()):
            checks.append(check("data", "OK" if dmax_d == last_session else "WARN",
                                f"panel {dmax_d}, last session {last_session} (market closed today)"))
        elif dmax_d == last_session:
            checks.append(check("data", "OK", f"panel through {dmax_d}"))
        else:
            checks.append(check("data", "FAIL", f"panel through {dmax_d}, last completed session {last_session}"))
    except Exception as e:                           # noqa: BLE001
        checks.append(check("data", "FAIL", f"cannot read {data_path.name}: {e}"))

    # signal file
    artifacts = artifacts or ROOT / "research" / "artifacts"
    files = sorted(glob.glob(str(artifacts / "signals_multi_factor_*.json")))
    if not files:
        checks.append(check("signals", "FAIL", "no signal file"))
    else:
        f = Path(files[-1])
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            dd = str(d.get("data_date", ""))[:10]
            gen = datetime.fromisoformat(d["generated_at"])
            age_h = (datetime.now() - gen).total_seconds() / 3600
            problems = []
            if dd != last_session.isoformat():
                problems.append(f"data_date {dd} != last session {last_session}")
            if age_h > SIGNAL_MAX_AGE_H:
                problems.append(f"{age_h:.0f}h old")
            if "selection_params" not in d:
                problems.append("no selection_params")
            if not cal.is_trading_day(now_et.date()) and problems and dd == last_session.isoformat():
                problems = [p for p in problems if "old" not in p]
            st = "OK" if not problems else ("WARN" if not cal.is_trading_day(now_et.date()) else "FAIL")
            checks.append(check("signals", st, f"{f.name}: data_date {dd}, {age_h:.0f}h old, "
                                                f"{d.get('n_stocks')} names, should_trade={d.get('should_trade')}"
                                                + (f"; {'; '.join(problems)}" if problems else "")))
        except Exception as e:                       # noqa: BLE001
            checks.append(check("signals", "FAIL", f"{f.name} unreadable: {e}"))

    # model release
    try:
        from factors.factor_definitions import FACTOR_GROUPS
        from factors.model_release import verify
        ok, msgs = verify(artifacts, required=list(FACTOR_GROUPS.keys()))
        st = "OK" if ok and not any("UNVERIFIED" in m for m in msgs) else ("WARN" if ok else "FAIL")
        checks.append(check("models", st, "; ".join(msgs)[:300]))
    except Exception as e:                           # noqa: BLE001
        checks.append(check("models", "FAIL", f"verify failed: {e}"))

    # book vs broker
    if broker is not None and registry is not None:
        try:
            from trading import execution as ex
            rep = ex.reconcile(broker, registry, dry_run=True)
            if rep.aborted:
                checks.append(check("book", "FAIL", f"reconcile aborted: {rep.reason}"))
            else:
                bits = []
                if rep.orphans:
                    bits.append(f"{len(rep.orphans)} orphan(s): " + ", ".join(f"{s} {sd} {bq:.0f}/{rq}" for s, sd, bq, rq in rep.orphans))
                if rep.never_filled:
                    bits.append(f"{len(rep.never_filled)} never-filled leg(s)")
                if rep.reduced or rep.removed:
                    bits.append(f"{len(rep.reduced)} reduced, {len(rep.removed)} removed pending")
                checks.append(check("book", "WARN" if bits else "OK",
                                    "; ".join(bits) if bits else f"registry matches broker ({registry.summary().get('n_open')} open tranches)"))
        except Exception as e:                       # noqa: BLE001
            checks.append(check("book", "FAIL", f"reconcile error: {e}"))
    else:
        checks.append(check("book", "SKIP", "no broker"))

    # halt
    try:
        from trading import halt
        hp = halt_path or halt.HALT_PATH
        st = halt.read(hp)
        checks.append(check("halt", "WARN" if st.get("halted") else "OK",
                            f"HALTED since {st.get('triggered_at')}: {st.get('reason')}" if st.get("halted") else "no halt"))
    except Exception as e:                           # noqa: BLE001
        checks.append(check("halt", "FAIL", str(e)))

    # failures today
    fl = failures_log or FAILURES_LOG
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        today_alt = datetime.now().strftime("%Y/%m/%d")
        n = 0
        if fl.exists():
            for line in fl.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith(today) or line.startswith(today_alt):
                    n += 1
        checks.append(check("failures", "WARN" if n else "OK", f"{n} FAILURES.log line(s) today"))
    except Exception as e:                           # noqa: BLE001
        checks.append(check("failures", "FAIL", str(e)))
    return checks


def overall(checks: list[dict]) -> str:
    sts = {c["status"] for c in checks}
    return "FAIL" if "FAIL" in sts else ("WARN" if "WARN" in sts else "OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nightly-rc", type=int, default=None)
    ap.add_argument("--no-broker", action="store_true")
    args = ap.parse_args(argv)
    from trading import trading_calendar as cal
    now_et = cal.eastern_now()
    broker = registry = None
    if not args.no_broker:
        try:
            import config_trading
            from trading.broker import AlpacaBroker
            from trading.tranche_registry import TrancheRegistry
            broker = AlpacaBroker()
            registry = TrancheRegistry(ROOT / config_trading.TRANCHE_REGISTRY_PATH)
        except Exception as e:                       # noqa: BLE001
            print(f"[health] broker unavailable for the book check: {e}")
    checks = run_checks(now_et, nightly_rc=args.nightly_rc, broker=broker, registry=registry)
    status = overall(checks)
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    HEALTH_JSON.write_text(json.dumps({"at": stamp, "status": status, "checks": checks}, indent=2), encoding="utf-8")
    line = f"{stamp} {status} | " + " | ".join(f"{c['name']}={c['status']}" for c in checks)
    with open(HEALTH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[health] {status}")
    for c in checks:
        print(f"  {c['status']:<5} {c['name']:<10} {c['detail']}")
    if status == "FAIL":
        bad = "; ".join(f"{c['name']}: {c['detail']}" for c in checks if c["status"] == "FAIL")
        with open(FAILURES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{stamp} HEALTH CHECK FAIL: {bad}\n")
        try:
            from trading.notify import notify
            notify("StockPredict health check FAIL", bad[:400])
        except Exception:                            # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(main())
