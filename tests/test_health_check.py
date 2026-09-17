"""run_health_check.run_checks: every check evaluated on synthetic inputs
(no SDK, no keys), the statuses it must produce, and the FAIL rollup."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "research"))

hc = pytest.importorskip("run_health_check")
from trading.broker import FakeBroker  # noqa: E402
from trading.trading_calendar import EASTERN  # noqa: E402
from trading.tranche_registry import TrancheRegistry, TranchePosition  # noqa: E402

NOW = datetime(2026, 9, 16, 21, 30, tzinfo=EASTERN)          # Wednesday evening: last completed session = 09-16
LAST = "2026-09-16"


def _env(tmp_path, data_last=LAST, signal_date=LAST, signal_age_h=1.0, with_params=True, manifest=True):
    data = tmp_path / "panel.parquet"
    pd.DataFrame({"date": pd.bdate_range(end=data_last, periods=3), "symbol": "AAA"}).to_parquet(data, index=False)
    art = tmp_path / "artifacts"
    art.mkdir()
    sig = {"generated_at": (datetime.now() - timedelta(hours=signal_age_h)).isoformat(),
           "data_date": f"{signal_date} 00:00:00-04:00", "should_trade": True, "n_stocks": 10, "stocks": []}
    if with_params:
        sig["selection_params"] = {"version": "x"}
    (art / f"signals_multi_factor_{signal_date.replace('-', '')}.json").write_text(json.dumps(sig), encoding="utf-8")
    from factors.factor_definitions import FACTOR_GROUPS
    from factors import model_release as mr
    for n in FACTOR_GROUPS:
        (art / mr.ensemble_file(n)).write_bytes(b"m" + n.encode())
    (art / "factor_weights.json").write_text("{}", encoding="utf-8")
    if manifest:
        mr.write_manifest(art, list(FACTOR_GROUPS))
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t1", "2026-09-10", 20)
    reg.add_position("t1", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t1_AAA"))
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    return data, art, b, reg


def _by(checks):
    return {c["name"]: c for c in checks}


def test_all_green_on_a_consistent_night(tmp_path):
    data, art, b, reg = _env(tmp_path)
    checks = hc.run_checks(NOW, nightly_rc=0, data_path=data, artifacts=art, broker=b, registry=reg,
                           halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log")
    st = _by(checks)
    assert {c["status"] for c in checks} <= {"OK"} , st
    assert hc.overall(checks) == "OK"


def test_each_failure_mode_is_named(tmp_path):
    data, art, b, reg = _env(tmp_path, data_last="2026-09-15", signal_date="2026-09-15", signal_age_h=40, with_params=False, manifest=False)
    b.pos = {"AAA": 10.0, "ZZZ": 5.0}                                   # orphan
    (tmp_path / "F.log").write_text(f"{datetime.now():%Y-%m-%d} 03:00 NIGHTLY TRADING RUN FAILED exit=1\n", encoding="utf-8")
    from trading import halt
    halt.set_halt("test", path=tmp_path / "halt.json")
    checks = hc.run_checks(NOW, nightly_rc=1, data_path=data, artifacts=art, broker=b, registry=reg,
                           halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log")
    st = _by(checks)
    assert st["nightly_rc"]["status"] == "FAIL"
    assert st["data"]["status"] == "FAIL" and "2026-09-15" in st["data"]["detail"]
    assert st["signals"]["status"] == "FAIL" and "data_date" in st["signals"]["detail"] and "old" in st["signals"]["detail"]
    assert st["models"]["status"] == "WARN" and "UNVERIFIED" in st["models"]["detail"]
    assert st["book"]["status"] == "WARN" and "orphan" in st["book"]["detail"] and "ZZZ" in st["book"]["detail"]
    assert st["halt"]["status"] == "WARN" and st["failures"]["status"] == "WARN" and "1 FAILURES" in st["failures"]["detail"]
    assert hc.overall(checks) == "FAIL"


def test_broker_outage_and_missing_broker(tmp_path):
    data, art, b, reg = _env(tmp_path)
    b.fail_next["positions"] = 1
    st = _by(hc.run_checks(NOW, nightly_rc=0, data_path=data, artifacts=art, broker=b, registry=reg,
                           halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log"))
    assert st["book"]["status"] == "FAIL" and "aborted" in st["book"]["detail"]
    st2 = _by(hc.run_checks(NOW, nightly_rc=None, data_path=data, artifacts=art, broker=None, registry=None,
                            halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log"))
    assert st2["book"]["status"] == "SKIP" and st2["nightly_rc"]["status"] == "SKIP"


def test_closed_market_day_is_a_warning_not_a_failure(tmp_path):
    # Labor Day evening: the panel and the signal file legitimately stop at Friday 09-04
    data, art, b, reg = _env(tmp_path, data_last="2026-09-04", signal_date="2026-09-04", signal_age_h=60)
    st = _by(hc.run_checks(datetime(2026, 9, 7, 21, 30, tzinfo=EASTERN), nightly_rc=0, data_path=data, artifacts=art,
                           broker=b, registry=reg, halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log"))
    assert st["data"]["status"] == "OK" and st["signals"]["status"] == "OK"
