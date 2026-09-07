"""Orchestrator-level pins for the phase-1 rewrite of run_staggered_trading,
driven by FakeBroker (no SDK, no keys):

  * a broker outage at reconcile aborts the cycle with the dedicated exit
    code and leaves the registry as it was
  * a partial close keeps the remainder registered and the tranche open
  * a halted account opens nothing; a daily-loss breach sets the halt and
    opens nothing; a normal day opens with the cumulative guards applied
"""

import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading.broker import FakeBroker  # noqa: E402
from trading import halt  # noqa: E402
from trading import trading_calendar as cal  # noqa: E402
from trading.tranche_registry import TrancheRegistry, TranchePosition  # noqa: E402

rst = pytest.importorskip("run_staggered_trading")

NOSLEEP = lambda s: None  # noqa: E731


@pytest.fixture(autouse=True)
def _isolated_halt(tmp_path, monkeypatch):
    monkeypatch.setattr(halt, "HALT_PATH", tmp_path / "halt.json")
    monkeypatch.setattr(rst, "FAILURES_LOG", tmp_path / "FAILURES.log")
    monkeypatch.setattr(rst, "EXEC_ARM_LOG", tmp_path / "exec_arms.csv")
    monkeypatch.setattr(rst.config_trading, "EXECUTION_AB_TEST", False)
    monkeypatch.setattr(rst, "compute_risk_levels",
                        lambda symbol, price, side: type("R", (), {"stop_price": price * 0.9, "take_price": price * 1.2,
                                                                     "stop_pct": 0.1, "take_pct": 0.2, "atr": None,
                                                                     "basis": "test"})())
    monkeypatch.setattr(rst, "atr_summary", lambda: "n/a")


def _due_registry(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t1", "2026-01-05", 20)                       # long past its close date
    reg.add_position("t1", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t1_AAA"))
    reg.add_position("t1", TranchePosition("SSS", "short", 4, 50.0, 200.0, "open_t1_SSS"))
    reg.save()
    return reg


def test_partial_close_keeps_remainder_and_tranche_open(tmp_path):
    reg = _due_registry(tmp_path)
    b = FakeBroker(prices={"AAA": 110, "SSS": 45})
    b.pos = {"AAA": 10.0, "SSS": -4.0}
    b.partial_fill["AAA"] = 0.5
    n = rst.close_due_tranches(b, reg, dry_run=False, sleep=NOSLEEP, timeout_s=0.2)
    t = reg.get("t1")
    assert n == 1 and t.status == "open"
    assert [(p.symbol, p.qty) for p in t.longs] == [("AAA", 5)] and t.shorts == []
    # second run closes the rest (the fake fills fully now) and the tranche closes
    b.partial_fill.clear()
    rst.close_due_tranches(b, reg, dry_run=False, sleep=NOSLEEP, timeout_s=0.2)
    assert reg.get("t1").status == "closed" and b.pos == {}


def _signals(data_date=None, generated_at=None):
    dd = data_date or cal.last_completed_session(cal.eastern_now()).isoformat()
    return {"should_trade": True, "n_stocks": 2, "data_date": f"{dd} 00:00:00-04:00",
            "generated_at": generated_at or datetime.now().isoformat(), "_file": f"signals_{dd}.json",
            "stocks": [{"symbol": "AAA", "prediction": 0.01, "position_pct": 50.0},
                       {"symbol": "SSS", "prediction": -0.01, "position_pct": 50.0}]}


def test_halted_account_opens_nothing(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=100_000, prices={"AAA": 100, "SSS": 50})
    halt.set_halt("test", path=halt.HALT_PATH)
    out = rst.open_new_tranche(b, reg, _signals(), dry_run=False, sleep=NOSLEEP)
    assert out is None and not [c for c in b.calls if c[0] == "submit"] and reg.summary()["n_open"] == 0


def test_daily_loss_breach_sets_halt_and_opens_nothing(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=96_000, last_equity=100_000, prices={"AAA": 100, "SSS": 50})
    out = rst.open_new_tranche(b, reg, _signals(), dry_run=False, sleep=NOSLEEP)
    assert out is None and halt.is_halted(halt.HALT_PATH)
    assert "daily loss" in halt.read(halt.HALT_PATH)["reason"]
    assert not [c for c in b.calls if c[0] == "submit"]


def test_normal_day_opens_with_cumulative_guards(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=100_000, last_equity=100_000, prices={"AAA": 100, "SSS": 50})
    b.pos = {"SSS": -38.0}                                        # 1,900 already short in SSS (cap 2% = 2,000)
    out = rst.open_new_tranche(b, reg, _signals(), dry_run=False, sleep=NOSLEEP)
    assert out is not None
    submits = [c for c in b.calls if c[0] == "submit"]
    assert [c[1] for c in submits] == ["AAA"]                     # SSS blocked by the single-name cap incl. existing
    t = reg.get(out)
    assert len(t.longs) == 1 and t.shorts == []
    assert submits[0][5] == f"open_{out}_AAA"


def test_broker_outage_at_reconcile_aborts_with_exit_code(tmp_path, monkeypatch):
    reg = _due_registry(tmp_path)
    b = FakeBroker(prices={"AAA": 100, "SSS": 50})
    b.fail_next["positions"] = 1
    rep = rst.reconcile_registry_with_broker(b, reg, dry_run=False)
    assert rep.aborted and reg.get("t1").longs[0].qty == 10
    # main() wires the abort to the exit code
    monkeypatch.setattr(rst.config_trading, "STRATEGY", "staggered")
    monkeypatch.setattr(rst.config_trading, "TRANCHE_REGISTRY_PATH", str(tmp_path / "reg.json"))
    import trading.broker as tb
    monkeypatch.setattr(tb, "AlpacaBroker", lambda: b)
    monkeypatch.setattr(rst, "TrancheRegistry", lambda p: reg)
    import types
    fake_trader_mod = types.ModuleType("alpaca_trader")
    fake_trader_mod.AlpacaAutoTrader = lambda: types.SimpleNamespace(get_current_price=lambda s: 100.0)
    monkeypatch.setitem(sys.modules, "alpaca_trader", fake_trader_mod)
    b.fail_next["positions"] = 1
    assert rst.main([]) == rst.EXIT_BROKER_UNAVAILABLE
    assert "BROKER UNAVAILABLE" in (tmp_path / "FAILURES.log").read_text(encoding="utf-8")


# phase 1b ------------------------------------------------------------------
def test_stale_or_duplicate_signals_open_nothing(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=100_000, last_equity=100_000, prices={"AAA": 100, "SSS": 50})
    fri_evening = datetime(2026, 9, 4, 17, 0, tzinfo=cal.EASTERN)
    # Thursday's data on Friday evening: stale
    out = rst.open_new_tranche(b, reg, _signals("2026-09-03"), dry_run=False, sleep=NOSLEEP, now_et=fri_evening)
    assert out is None and not [c for c in b.calls if c[0] == "submit"]
    assert "NOT FRESH" in (tmp_path / "FAILURES.log").read_text(encoding="utf-8")
    # Friday's data on Friday evening: fresh -> opens
    out = rst.open_new_tranche(b, reg, _signals("2026-09-04"), dry_run=False, sleep=NOSLEEP, now_et=fri_evening)
    assert out is not None and reg.get(out).notes == "data_date=2026-09-04"
    # Labor Day evening: Friday's data is still the last completed session, but
    # a tranche already used it -> duplicate, nothing opened
    labor_day = datetime(2026, 9, 7, 17, 0, tzinfo=cal.EASTERN)
    reg2 = TrancheRegistry(tmp_path / "reg.json")
    n_before = len([c for c in b.calls if c[0] == "submit"])
    out = rst.open_new_tranche(b, reg2, _signals("2026-09-04"), dry_run=False, sleep=NOSLEEP, now_et=labor_day)
    assert out is None and len([c for c in b.calls if c[0] == "submit"]) == n_before
    # a file generated two days ago is stale even with the right data_date
    old = (datetime.now() - timedelta(hours=50)).isoformat()
    ok, why = rst.signals_fresh(_signals("2026-09-04", generated_at=old), TrancheRegistry(tmp_path / "r2.json"), fri_evening)
    assert not ok and "generated" in why


def test_legacy_registry_data_dates_are_recognised(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t_legacy", "2026-09-04", 20, signal_file="2026-09-04 00:00:00-04:00")
    assert rst._tranche_data_dates(reg) == {"2026-09-04"}


def test_calendar_gate_uses_broker_and_logs_mismatch(tmp_path):
    b = FakeBroker()
    assert rst.calendar_gate(b, datetime(2026, 9, 8, 16, 30, tzinfo=cal.EASTERN))
    assert not rst.calendar_gate(b, datetime(2026, 9, 7, 16, 30, tzinfo=cal.EASTERN))     # Labor Day
    assert not (tmp_path / "FAILURES.log").exists()
    b2 = FakeBroker(holidays={date(2026, 9, 8)})                                            # unscheduled closure
    assert not rst.calendar_gate(b2, datetime(2026, 9, 8, 16, 30, tzinfo=cal.EASTERN))
    assert "CALENDAR MISMATCH" in (tmp_path / "FAILURES.log").read_text(encoding="utf-8")
    b3 = FakeBroker(); b3.fail_next["calendar"] = 1
    assert rst.calendar_gate(b3, datetime(2026, 9, 8, 16, 30, tzinfo=cal.EASTERN))         # built-in fallback


def test_market_closed_day_reconciles_only(tmp_path, monkeypatch):
    reg = _due_registry(tmp_path)
    b = FakeBroker(prices={"AAA": 100, "SSS": 50})
    b.pos = {"AAA": 10.0, "SSS": -4.0}
    monkeypatch.setattr(rst.config_trading, "STRATEGY", "staggered")
    import trading.broker as tb
    monkeypatch.setattr(tb, "AlpacaBroker", lambda: b)
    monkeypatch.setattr(rst, "TrancheRegistry", lambda p: reg)
    mod = types.ModuleType("alpaca_trader")
    mod.AlpacaAutoTrader = lambda: types.SimpleNamespace(get_current_price=lambda s: 100.0)
    monkeypatch.setitem(sys.modules, "alpaca_trader", mod)
    monkeypatch.setattr(rst, "_now_et", lambda: datetime(2026, 9, 7, 16, 30, tzinfo=cal.EASTERN))

    def boom(*a, **k):
        raise AssertionError("must not run on a closed day")
    monkeypatch.setattr(rst, "close_due_tranches", boom)
    monkeypatch.setattr(rst, "load_today_signals", boom)
    monkeypatch.setattr(rst.time, "sleep", NOSLEEP)
    assert rst.main([]) == 0
    assert reg.get("t1").status == "open" and not [c for c in b.calls if c[0] == "submit"]


def test_daily_loss_breach_notifies(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=96_000, last_equity=100_000, prices={"AAA": 100, "SSS": 50})
    assert rst.open_new_tranche(b, reg, _signals(), dry_run=False, sleep=NOSLEEP) is None
    assert halt._TEST_SENT and "daily loss" in halt._TEST_SENT[-1][1]


def test_shared_symbol_close_keeps_other_tranche_protected(tmp_path):
    from trading.broker import LIVE
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t_old", "2026-01-05", 20)
    reg.add_tranche("t_new", (date.today() - timedelta(days=1)).isoformat(), 20)
    b = FakeBroker(prices={"AAA": 100})
    old = b.submit("AAA", 10, "buy", bracket=(90.0, 120.0), client_order_id="open_t_old_AAA", tif="gtc")
    new = b.submit("AAA", 6, "buy", bracket=(95.0, 125.0), client_order_id="open_t_new_AAA", tif="gtc")
    for _ in range(3):
        b.open_orders()
    reg.add_position("t_old", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t_old_AAA"))
    reg.add_position("t_new", TranchePosition("AAA", "long", 6, 100.0, 600.0, "open_t_new_AAA"))
    reg.save()
    n = rst.close_due_tranches(b, reg, dry_run=False, sleep=NOSLEEP, timeout_s=0.5)
    assert n == 1 and reg.get("t_old").status == "closed" and reg.get("t_new").longs[0].qty == 6
    assert b.pos == {"AAA": 6.0}
    assert all(b.orders[l].status in LIVE for l in new.legs) and all(b.orders[l].status == "canceled" for l in old.legs)
