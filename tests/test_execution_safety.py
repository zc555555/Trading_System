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
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading.broker import BrokerError, FakeBroker  # noqa: E402
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
