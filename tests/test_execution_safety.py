"""Phase-1 execution safety (2026-09 review P0s), all against FakeBroker so
they run without the alpaca SDK or keys:

  1. a failed positions query ABORTS reconciliation -- nothing is deleted
  2. broker holding fewer shares reduces the oldest leg, never deletes a
     whole leg; broker holding more / unknown symbol is an orphan report
  3. close_leg returns FILLED shares: a partial fill leaves the remainder
     registered, a queued order is not "success", a retry after a dead
     attempt reuses the client id with a suffix (no duplicate orders)
  4. entry guards count existing positions AND pending entry orders per
     symbol and in aggregate (a second short in the same name is blocked)
  5. the halt state persists and blocks entries; the breaker flattens and
     cancels everything
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading.broker import BrokerError, FakeBroker, LIVE as LIVE_STATES  # noqa: E402
from trading import execution as ex  # noqa: E402
from trading import halt  # noqa: E402
from trading.tranche_registry import TrancheRegistry, TranchePosition  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731


def _registry(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t_old", "2026-08-01", 20)
    reg.add_position("t_old", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t_old_AAA"))
    reg.add_position("t_old", TranchePosition("SSS", "short", 8, 50.0, 400.0, "open_t_old_SSS"))
    reg.add_tranche("t_new", "2026-08-20", 20)
    reg.add_position("t_new", TranchePosition("AAA", "long", 6, 110.0, 660.0, "open_t_new_AAA"))
    reg.save()
    return reg


def _legs(reg, tid):
    t = reg.get(tid)
    return {(p.symbol, "long"): p.qty for p in t.longs} | {(p.symbol, "short"): p.qty for p in t.shorts}


# 1 ------------------------------------------------------------------------
def test_positions_failure_aborts_reconcile_without_touching_the_registry(tmp_path):
    reg = _registry(tmp_path)
    b = FakeBroker(prices={"AAA": 100, "SSS": 50})
    b.fail_next["positions"] = 1
    rep = ex.reconcile(b, reg)
    assert rep.aborted and "unavailable" in rep.reason
    assert _legs(reg, "t_old") == {("AAA", "long"): 10, ("SSS", "short"): 8}
    assert _legs(reg, "t_new") == {("AAA", "long"): 6}


# 2 ------------------------------------------------------------------------
def test_reconcile_reduces_oldest_leg_and_reports_orphans(tmp_path):
    reg = _registry(tmp_path)
    b = FakeBroker(prices={"AAA": 100, "SSS": 50, "ZZZ": 10})
    b.pos = {"AAA": 12.0, "SSS": -8.0, "ZZZ": 5.0}      # 4 AAA shares gone (16 registered), ZZZ unknown
    rep = ex.reconcile(b, reg)
    assert not rep.aborted
    assert rep.reduced == [("t_old", "AAA", "long", 10, 6)] and rep.removed == []
    assert _legs(reg, "t_old") == {("AAA", "long"): 6, ("SSS", "short"): 8}
    assert _legs(reg, "t_new") == {("AAA", "long"): 6}
    assert ("ZZZ", "long", 5.0, 0) in rep.orphans
    # broker holds nothing of AAA -> both legs removed, oldest first
    b.pos = {"SSS": -8.0}
    rep = ex.reconcile(b, reg)
    assert [r[:3] for r in rep.removed] == [("t_old", "AAA", "long"), ("t_new", "AAA", "long")]
    assert reg.get("t_new").status == "closed"


# 3 ------------------------------------------------------------------------
def test_close_leg_reports_filled_not_accepted_and_keeps_remainder(tmp_path):
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    b.partial_fill["AAA"] = 0.6                          # only 6 of 10 fill
    res = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=2, poll_s=0.01, sleep=NOSLEEP)
    assert res.filled == 6 and not res.complete and res.status == "partially_filled"
    reg = TrancheRegistry(tmp_path / "r.json")
    reg.add_tranche("t", "2026-08-01", 20)
    reg.add_position("t", TranchePosition("AAA", "long", 10, 100.0, 1000.0, "open_t_AAA"))
    reg.reduce_position("t", "AAA", "long", res.filled)
    assert reg.get("t").longs[0].qty == 4 and reg.get("t").status == "open"


def test_close_leg_waits_on_queued_order_instead_of_duplicating(tmp_path):
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    b.fill_delay_polls = 10**9                           # queued (e.g. after the close): never fills in this test
    res = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=0.05, poll_s=0.01, sleep=NOSLEEP)
    assert res.filled == 0 and res.status in ("accepted", "new") and "live" in res.note
    submits = [c for c in b.calls if c[0] == "submit"]
    assert len(submits) == 1
    # second run: the order is still live -> waited on, NOT resubmitted
    res2 = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=0.05, poll_s=0.01, sleep=NOSLEEP)
    assert len([c for c in b.calls if c[0] == "submit"]) == 1 and res2.filled == 0
    # once it fills, the same call credits the fill without a new order
    b.fill_delay_polls = 0
    b._pending_polls.clear()
    res3 = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=1, poll_s=0.01, sleep=NOSLEEP)
    assert res3.complete and len([c for c in b.calls if c[0] == "submit"]) == 1


def test_close_leg_retries_a_rejected_attempt_with_a_suffixed_id():
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    b.reject["AAA"] = "insufficient qty"
    res = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=0.05, poll_s=0.01, sleep=NOSLEEP, max_attempts=2)
    ids = [c[5] for c in b.calls if c[0] == "submit"]
    assert ids == ["close_t_AAA", "close_t_AAA_r2"] and res.filled == 0 and res.status == "rejected"


def test_close_cancels_resting_brackets_and_waits_for_them():
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    b.cancel_delay_polls = 2
    stop = b.submit("AAA", 10, "sell", kind="limit", limit_price=90.0, client_order_id="open_t_AAA_stop")
    res = ex.close_leg(b, "AAA", 10, "sell", "close_t_AAA", timeout_s=2, poll_s=0.01, sleep=NOSLEEP)
    assert res.complete and b.orders[stop.id].status == "canceled"
    order = [c for c in b.calls if c[0] == "submit" and c[5] == "close_t_AAA"]
    assert len(order) == 1


# 4 ------------------------------------------------------------------------
def test_entry_guards_count_existing_and_pending_exposure():
    b = FakeBroker(equity=100_000, prices={"SSS": 50, "AAA": 100})
    b.pos = {"SSS": -30.0}                               # 1,500 short already
    b.submit("SSS", 10, "sell", kind="limit", limit_price=50.0, client_order_id="open_t_SSS")   # pending 500 short
    snap = ex.exposure(b)
    assert snap.pending_orders == 1 and snap.short == 2000 and snap.by_symbol_short["SSS"] == 2000
    ok, why = ex.entry_allowed(snap, "SSS", "short", 100, 100_000, short_single_max_pct=2.0, short_gross_max_pct=25.0)
    assert not ok and "single short" in why                # 2,100 > 2% of 100,000
    ok, _ = ex.entry_allowed(snap, "TTT", "short", 1900, 100_000, 2.0, 25.0)
    assert ok
    ex.apply_entry(snap, "TTT", "short", 1900)
    ok, why = ex.entry_allowed(snap, "TTT", "short", 200, 100_000, 2.0, 25.0)
    assert not ok                                        # cumulative within the run
    ok, why = ex.entry_allowed(snap, "AAA", "long", 97_000, 100_000, 2.0, 25.0)
    assert not ok and "no leverage" in why
    b.fail_next["positions"] = 1
    with pytest.raises(BrokerError):
        ex.exposure(b)


# 5 ------------------------------------------------------------------------
def test_halt_persists_and_breaker_flattens_everything(tmp_path):
    path = tmp_path / "halt.json"
    assert not halt.is_halted(path)
    assert halt.daily_loss_breached(96_900, 100_000, 0.03) and not halt.daily_loss_breached(97_100, 100_000, 0.03)
    halt.set_halt("daily loss -3.1%", equity=96_900, path=path)
    assert halt.is_halted(path) and halt.read(path)["reason"].startswith("daily loss")
    b = FakeBroker(prices={"AAA": 100, "SSS": 50})
    b.pos = {"AAA": 10.0, "SSS": -4.0}
    b.submit("BBB", 5, "buy", kind="limit", limit_price=10.0, client_order_id="open_t_BBB")
    rep = ex.flatten_all(b, sleep=NOSLEEP, timeout_s=2, poll_s=0.01)
    assert rep["canceled"] >= 1 and not b.pos and rep["unfilled"] == [] and len(rep["closed"]) == 2
    halt.clear("test", path=path)
    assert not halt.is_halted(path)
    path.write_text("{not json", encoding="utf-8")
    assert halt.is_halted(path)                          # unreadable state fails safe


# 6 (phase 1b) ------------------------------------------------------------
def _fill_all(b):
    for _ in range(3):
        b.open_orders()


def test_close_cancels_only_this_legs_bracket_children_when_symbol_is_shared():
    b = FakeBroker(prices={"AAA": 100})
    old = b.submit("AAA", 10, "buy", bracket=(90.0, 120.0), client_order_id="open_t_old_AAA", tif="gtc")
    new = b.submit("AAA", 6, "buy", bracket=(95.0, 125.0), client_order_id="open_t_new_AAA", tif="gtc")
    _fill_all(b)
    assert b.pos == {"AAA": 16.0} and len(old.legs) == 2 and len(new.legs) == 2
    assert b.held_qty("AAA", "sell") == 16
    res = ex.close_leg(b, "AAA", 10, "sell", "close_t_old_AAA", timeout_s=2, poll_s=0.01, sleep=NOSLEEP,
                       entry_client_id="open_t_old_AAA", symbol_shared=True)
    assert res.complete and b.pos == {"AAA": 6.0}
    assert all(b.orders[l].status == "canceled" for l in old.legs)
    assert all(b.orders[l].status in LIVE_STATES for l in new.legs)       # t_new keeps its stop and take
    assert b.held_qty("AAA", "sell") == 6


def test_shared_symbol_with_unknown_entry_never_strips_the_other_tranche():
    b = FakeBroker(prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    new = b.submit("AAA", 6, "buy", bracket=(95.0, 125.0), client_order_id="open_t_new_AAA", tif="gtc")
    _fill_all(b)
    # legacy leg: entry order unknown, symbol shared -> nothing cancelled, close is
    # attempted and rejected by the held shares -> leg kept for a human
    res = ex.close_leg(b, "AAA", 16, "sell", "close_t_old_AAA", timeout_s=0.5, poll_s=0.01, sleep=NOSLEEP,
                       entry_client_id=None, symbol_shared=True, max_attempts=2)
    assert res.filled == 0 and res.status == "rejected"
    assert all(b.orders[l].status in LIVE_STATES for l in new.legs)
    assert not [c for c in b.calls if c[0] == "cancel"]
    # same leg, symbol NOT shared -> the symbol-wide cancel is allowed (old behaviour)
    res = ex.close_leg(b, "AAA", 16, "sell", "close_t_old2_AAA", timeout_s=2, poll_s=0.01, sleep=NOSLEEP,
                       entry_client_id=None, symbol_shared=False)
    assert res.complete and b.pos == {}


def test_fake_broker_rejects_closes_beyond_free_shares():
    b = FakeBroker(prices={"AAA": 100})
    b.submit("AAA", 10, "buy", bracket=(90.0, 120.0), client_order_id="e1")
    _fill_all(b)
    o = b.submit("AAA", 10, "sell", client_order_id="c1")
    assert o.status == "rejected" and any(c[0] == "reject" for c in b.calls)
    b.cancel(b.orders["fake-1"].legs[0])                                   # OCO: sibling goes too
    assert b.held_qty("AAA", "sell") == 0
    assert b.submit("AAA", 10, "sell", client_order_id="c2").status == "accepted"


def test_halt_set_notifies_operator(tmp_path):
    halt.set_halt("daily loss -3.5%", equity=96_500, path=tmp_path / "h.json")
    assert halt._TEST_SENT and "HALT" in halt._TEST_SENT[-1][0] and "daily loss" in halt._TEST_SENT[-1][1]


# 7 (phase 1b): pending entries are legs in flight, stale limit entries expire --
def test_reconcile_keeps_legs_whose_entry_is_still_in_flight(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t", "2026-09-04", 20)
    reg.add_position("t", TranchePosition("BAX", "long", 20, 27.0, 540.0, "open_t_BAX"))
    reg.add_position("t", TranchePosition("ROP", "short", 1, 400.0, 400.0, "open_t_ROP"))
    reg.save()
    b = FakeBroker(prices={"BAX": 27, "ROP": 400})
    b.fill_delay_polls = 10**9                                   # queued after the close: nothing fills
    b.submit("BAX", 20, "buy", kind="market", bracket=(24.0, 33.0), client_order_id="open_t_BAX", tif="gtc")
    b.submit("ROP", 1, "sell", kind="limit", limit_price=400.0, bracket=(440.0, 320.0), client_order_id="open_t_ROP", tif="gtc")
    rep = ex.reconcile(b, reg)
    assert not rep.aborted and rep.removed == [] and rep.reduced == []
    assert sorted(x[1] for x in rep.pending) == ["BAX", "ROP"]
    assert reg.get("t").longs[0].qty == 20 and reg.get("t").shorts[0].qty == 1
    # the orders query failing aborts too (in-flight legs would look vanished)
    b.fail_next["open_orders"] = 1
    assert ex.reconcile(b, reg).aborted


def test_expire_stale_limit_entries_but_not_queued_market_orders(tmp_path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t", "2026-09-03", 20)
    reg.add_position("t", TranchePosition("BAX", "long", 20, 27.0, 540.0, "open_t_BAX"))
    reg.add_position("t", TranchePosition("ROP", "short", 4, 400.0, 1600.0, "open_t_ROP"))
    reg.add_position("t", TranchePosition("NEW", "long", 3, 10.0, 30.0, "open_t_NEW"))
    reg.save()
    b = FakeBroker(prices={"BAX": 27, "ROP": 400, "NEW": 10})
    b.fill_delay_polls = 10**9
    b.clock = datetime(2026, 9, 3, 20, 30, tzinfo=timezone.utc)        # placed Thursday evening
    b.submit("BAX", 20, "buy", kind="market", client_order_id="open_t_BAX", tif="gtc")
    rop = b.submit("ROP", 4, "sell", kind="limit", limit_price=400.0, client_order_id="open_t_ROP", tif="gtc")
    rop.filled_qty = 1.0; rop.status = "partially_filled"; b.pos["ROP"] = -1.0
    b.clock = datetime(2026, 9, 4, 20, 30, tzinfo=timezone.utc)        # placed Friday evening (fresh)
    b.submit("NEW", 3, "buy", kind="limit", limit_price=10.0, client_order_id="open_t_NEW", tif="gtc")
    stale_before = datetime(2026, 9, 4, 9, 30, tzinfo=timezone(timedelta(hours=-4)))   # Friday's open, ET
    dry = ex.expire_stale_entries(b, reg, stale_before, dry_run=True, sleep=NOSLEEP, poll_s=0.01)
    assert [x[1] for x in dry.expired] == ["ROP"] and not [c for c in b.calls if c[0] == "cancel"]
    rep = ex.expire_stale_entries(b, reg, stale_before, dry_run=False, sleep=NOSLEEP, poll_s=0.01)
    assert [(x[1], x[3], x[4]) for x in rep.expired] == [("ROP", 4, 1)]
    t = reg.get("t")
    assert [(p.symbol, p.qty) for p in t.longs] == [("BAX", 20), ("NEW", 3)]     # market + fresh limit untouched
    assert [(p.symbol, p.qty) for p in t.shorts] == [("ROP", 1)]                # shrunk to the filled part
    assert b.orders[rop.id].status == "canceled"
