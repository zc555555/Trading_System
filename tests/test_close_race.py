"""Pins the fix for the 2026-08-26 close race: after cancelling resting
bracket legs, the trader must wait until the broker reports no open
orders for the symbol before the closing order is submitted."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

# the live modules need the alpaca SDK and the git-ignored key file; CI
# has neither, so these tests run only where the trading stack exists
pytest.importorskip("alpaca")
if not (Path(__file__).resolve().parent.parent / "config_alpaca.py").exists():
    pytest.skip("config_alpaca.py not present", allow_module_level=True)


class _FakeClient:
    """Cancellation takes two polls to be processed, like the real broker."""

    def __init__(self):
        self.polls = 0
        self.cancelled = []

    def get_orders(self, req):
        self.polls += 1
        # first call (listing), second call (poll #1): still open; then empty
        return [SimpleNamespace(id="o1")] if self.polls <= 2 else []

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)


def _trader_with(client):
    import alpaca_trader
    t = object.__new__(alpaca_trader.AlpacaAutoTrader)
    t.trading_client = client
    return t


def test_cancel_waits_for_broker_to_process(monkeypatch):
    import alpaca_trader
    monkeypatch.setattr(alpaca_trader, "GetOrdersRequest", lambda **kw: kw, raising=False)
    monkeypatch.setattr(alpaca_trader, "QueryOrderStatus", SimpleNamespace(OPEN="open"), raising=False)
    client = _FakeClient()
    t = _trader_with(client)
    n = t.cancel_open_orders("TRV")
    assert n == 1 and client.cancelled == ["o1"]
    # listing + at least one poll that still saw the order + the empty poll
    assert client.polls >= 3


# test_close_path_retries_once was retired on 2026-09-07: the close path moved to
# trading/execution.close_leg, whose retry/idempotency rules are pinned in
# tests/test_execution_safety.py.
