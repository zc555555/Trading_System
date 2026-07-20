"""Tranche registry state-machine tests (ported from the P0 verification)."""

from pathlib import Path

from trading.tranche_registry import TrancheRegistry, TranchePosition


def _pos(symbol, side, qty=10, price=100.0):
    return TranchePosition(symbol=symbol, side=side, qty=qty, entry_price=price,
                           entry_amount_usd=qty * price,
                           client_order_id=f"o_{symbol}")


def test_incremental_add_position_persists(tmp_path: Path):
    reg_path = tmp_path / "reg.json"
    reg = TrancheRegistry(reg_path)
    reg.add_tranche("t1", open_date="2026-07-20", hold_days=5)
    reg.add_position("t1", _pos("AAA", "long"))
    reg.save()
    reg.add_position("t1", _pos("BBB", "short", qty=5, price=50.0))
    reg.save()

    reloaded = TrancheRegistry(reg_path).get("t1")
    assert len(reloaded.longs) == 1
    assert len(reloaded.shorts) == 1


def test_partial_close_keeps_tranche_open_for_retry(tmp_path: Path):
    reg_path = tmp_path / "reg.json"
    reg = TrancheRegistry(reg_path)
    reg.add_tranche("t1", open_date="2026-07-20", hold_days=5)
    reg.add_position("t1", _pos("AAA", "long"))
    reg.add_position("t1", _pos("BBB", "short"))
    reg.save()

    reg.remove_position("t1", "AAA", "long", reason="scheduled_close")
    reg.save()

    t1 = TrancheRegistry(reg_path).get("t1")
    assert t1.status == "open"
    assert len(t1.longs) == 0 and len(t1.shorts) == 1
    # still due -> retried next run
    due = TrancheRegistry(reg_path).tranches_due_to_close("2026-07-28")
    assert any(t.id == "t1" for t in due)


def test_emptied_tranche_closes_with_given_reason(tmp_path: Path):
    reg_path = tmp_path / "reg.json"
    reg = TrancheRegistry(reg_path)
    reg.add_tranche("t1", open_date="2026-07-20", hold_days=5)
    reg.add_position("t1", _pos("AAA", "long"))
    reg.remove_position("t1", "AAA", "long", reason="scheduled_close")
    reg.mark_closed("t1", 12.5, reason="scheduled_close")
    reg.save()

    t1 = TrancheRegistry(reg_path).get("t1")
    assert t1.status == "closed"
    assert t1.close_reason == "scheduled_close"
    assert t1.realized_pnl == 12.5


def test_duplicate_tranche_id_rejected(tmp_path: Path):
    reg = TrancheRegistry(tmp_path / "reg.json")
    reg.add_tranche("t1", open_date="2026-07-20", hold_days=5)
    try:
        reg.add_tranche("t1", open_date="2026-07-21", hold_days=5)
        raised = False
    except ValueError:
        raised = True
    assert raised
