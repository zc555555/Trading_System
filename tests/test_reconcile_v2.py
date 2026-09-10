"""Reconcile v2 (2026-09-10 audit): the registry converges from OUR OWN entry
orders, idempotently.

  * an entry that ended without filling drops its leg (the 09-03 ORCL/ROP/AMGN
    phantoms); a partially filled one shrinks to what filled
  * a bracket exit (stop / take) is charged to the leg it closed, not to the
    oldest tranche holding the symbol (the HPE mis-attribution)
  * a filled entry of an OPEN tranche whose leg the registry lost is restored
    from the fill (08-19 ORCL/AMGN, 08-26 ORCL/ROP)
  * running twice changes nothing the second time
  * the orchestrator keys tranches by the signal's data_date, so a make-up run
    before the open and the regular run after the close both trade
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading.broker import FakeBroker  # noqa: E402
from trading import execution as ex  # noqa: E402
from trading.tranche_registry import TrancheRegistry, TranchePosition  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731


def _fill_all(b):
    b.open_orders()                                   # one poll: market entries fill


def _book(tmp_path):
    """Two tranches; every entry placed through the fake broker with our ids."""
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(prices={"AAA": 100, "BBB": 50, "CCC": 20})
    b.clock = datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc)
    reg.add_tranche("t_old", "2026-08-19", 20)
    for sym, q, side in (("AAA", 10, "buy"), ("BBB", 4, "buy")):
        b.submit(sym, q, side, bracket=(0.9 * b.prices[sym], 1.2 * b.prices[sym]), client_order_id=f"open_t_old_{sym}", tif="gtc")
    b.clock = datetime(2026, 8, 27, 8, 0, tzinfo=timezone.utc)
    reg.add_tranche("t_new", "2026-08-26", 20)
    b.submit("AAA", 6, "buy", bracket=(90.0, 120.0), client_order_id="open_t_new_AAA", tif="gtc")
    _fill_all(b)
    reg.add_position("t_old", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t_old_AAA"))
    reg.add_position("t_old", TranchePosition("BBB", "long", 4, 50.0, 200.0, "open_t_old_BBB"))
    reg.add_position("t_new", TranchePosition("AAA", "long", 6, 100.0, 600.0, "open_t_new_AAA"))
    reg.save()
    return reg, b


def _legs(reg, tid):
    t = reg.get(tid)
    return {(p.symbol, "long"): p.qty for p in t.longs} | {(p.symbol, "short"): p.qty for p in t.shorts}


def test_never_filled_entry_drops_its_leg_even_when_symbol_total_is_covered(tmp_path):
    reg, b = _book(tmp_path)
    # a third tranche registers a 5-share AAA leg whose limit entry was cancelled unfilled
    b.clock = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)
    reg.add_tranche("t_ghost", "2026-09-03", 20)
    o = b.submit("AAA", 5, "buy", kind="limit", limit_price=95.0, client_order_id="open_t_ghost_AAA", tif="gtc")
    b.cancel(o.id)
    reg.add_position("t_ghost", TranchePosition("AAA", "long", 5, 95.0, 475.0, "open_t_ghost_AAA"))
    # and the registry LOST t_old's AAA leg -> symbol total 11 vs broker 16: the old
    # reconcile saw "broker holds more" and reported an orphan, never the phantom
    reg.remove_position("t_old", "AAA", "long", reason="lost")
    rep = ex.reconcile(b, reg)
    assert not rep.aborted
    assert rep.never_filled == [("t_ghost", "AAA", "long", 5, 0)]
    assert rep.restored == [("t_old", "AAA", "long", 10, 100.0)]
    assert _legs(reg, "t_ghost") == {} and reg.get("t_ghost").status == "closed"
    assert _legs(reg, "t_old") == {("AAA", "long"): 10, ("BBB", "long"): 4}
    assert rep.orphans == []
    # restored leg carries the bracket prices and our client id
    p = [p for p in reg.get("t_old").longs if p.symbol == "AAA"][0]
    assert p.client_order_id == "open_t_old_AAA" and p.stop_price == 90.0 and p.take_price == 120.0


def test_bracket_exit_is_charged_to_its_own_leg_not_oldest_first(tmp_path):
    reg, b = _book(tmp_path)
    # t_new's stop fires: broker AAA 16 -> 10. The oldest-first rule alone would
    # have cut t_old's 10-share leg to 4 and left t_new's dead leg at 6.
    b.trigger_exit(b.get_order_by_client_id("open_t_new_AAA").id, which="stop")
    assert b.pos["AAA"] == 10
    rep = ex.reconcile(b, reg)
    assert rep.exited == [("t_new", "AAA", "long", 6, 90.0)] and rep.reduced == [] and rep.removed == []
    assert _legs(reg, "t_old") == {("AAA", "long"): 10, ("BBB", "long"): 4}
    assert _legs(reg, "t_new") == {} and reg.get("t_new").status == "closed"
    # idempotent: a second pass sees nothing to do
    rep2 = ex.reconcile(b, reg)
    assert not rep2.changed and rep2.orphans == []


def test_partial_scheduled_close_is_not_mistaken_for_an_exit(tmp_path):
    reg, b = _book(tmp_path)
    # a scheduled partial close took 4 of t_old's 10 AAA (leg already reduced by close_leg)
    res = ex.close_leg(b, "AAA", 4, "sell", "close_t_old_AAA", timeout_s=1, poll_s=0.01, sleep=NOSLEEP,
                       entry_client_id="open_t_old_AAA", symbol_shared=True)
    assert res.complete
    reg.reduce_position("t_old", "AAA", "long", 4, reason="partial_close")
    rep = ex.reconcile(b, reg)
    assert not rep.changed and rep.orphans == []
    assert _legs(reg, "t_old") == {("AAA", "long"): 6, ("BBB", "long"): 4}


def test_reconcile_still_aborts_when_history_is_unavailable(tmp_path):
    reg, b = _book(tmp_path)
    b.fail_next["orders_since"] = 1
    rep = ex.reconcile(b, reg)
    assert rep.aborted and _legs(reg, "t_old") == {("AAA", "long"): 10, ("BBB", "long"): 4}


# orchestrator: tranche id per data session --------------------------------
rst = pytest.importorskip("run_staggered_trading")
from trading import halt  # noqa: E402
from trading import trading_calendar as cal  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(halt, "HALT_PATH", tmp_path / "halt.json")
    monkeypatch.setattr(rst, "FAILURES_LOG", tmp_path / "FAILURES.log")
    monkeypatch.setattr(rst, "EXEC_ARM_LOG", tmp_path / "exec_arms.csv")
    monkeypatch.setattr(rst.config_trading, "EXECUTION_AB_TEST", False)
    monkeypatch.setattr(rst, "compute_risk_levels",
                        lambda symbol, price, side: type("R", (), {"stop_price": price * 0.9, "take_price": price * 1.2,
                                                                     "stop_pct": 0.1, "take_pct": 0.2, "atr": None,
                                                                     "basis": "test"})())
    monkeypatch.setattr(rst, "atr_summary", lambda: "n/a")


def _signals(dd):
    return {"should_trade": True, "n_stocks": 1, "data_date": f"{dd} 00:00:00-04:00",
            "generated_at": datetime.now().isoformat(), "_file": f"signals_{dd}.json",
            "stocks": [{"symbol": "AAA", "prediction": 0.01, "position_pct": 100.0}]}


def test_make_up_run_and_regular_run_open_one_tranche_per_data_session(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    b = FakeBroker(equity=100_000, last_equity=100_000, prices={"AAA": 100})
    # 05:00 ET on Thursday 09-10: missed Wednesday evening -> trades Wednesday's data
    early = datetime(2026, 9, 10, 5, 0, tzinfo=cal.EASTERN)
    t1 = rst.open_new_tranche(b, reg, _signals("2026-09-09"), dry_run=False, sleep=NOSLEEP, now_et=early)
    assert t1 == "tranche_20260909" and reg.get(t1).notes == "data_date=2026-09-09"
    # 17:00 ET the same day: Thursday's data -> a second tranche, not "already ran today"
    late = datetime(2026, 9, 10, 17, 0, tzinfo=cal.EASTERN)
    t2 = rst.open_new_tranche(b, reg, _signals("2026-09-10"), dry_run=False, sleep=NOSLEEP, now_et=late)
    assert t2 == "tranche_20260910"
    # the same data twice is still refused
    assert rst.open_new_tranche(b, reg, _signals("2026-09-10"), dry_run=False, sleep=NOSLEEP, now_et=late) is None


def test_existing_calendar_keyed_tranche_gets_a_suffixed_sibling(tmp_path):
    """The live registry already holds tranche_20260910 carrying 09-09 data (the
    make-up run of 2026-09-10 05:02 ET); the evening run must not be blocked."""
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("tranche_20260910", "2026-09-10", 20, notes="data_date=2026-09-09")
    reg.save()
    b = FakeBroker(equity=100_000, last_equity=100_000, prices={"AAA": 100})
    late = datetime(2026, 9, 10, 17, 0, tzinfo=cal.EASTERN)
    t2 = rst.open_new_tranche(b, reg, _signals("2026-09-10"), dry_run=False, sleep=NOSLEEP, now_et=late)
    assert t2 == "tranche_20260910_2" and reg.get(t2).notes == "data_date=2026-09-10"
    assert [c[5] for c in b.calls if c[0] == "submit"] == ["open_tranche_20260910_2_AAA"]
