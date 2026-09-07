"""Execution-safety logic shared by the orchestrator and the monitor
(phase 1 of the 2026-09 review). Pure functions over a Broker (see
trading/broker.py) and a TrancheRegistry, so every path is testable with
FakeBroker and no SDK.

  wait_for_terminal   poll one order until filled/canceled/rejected or timeout
  cancel_symbol_orders cancel resting orders for a symbol and WAIT until the
                      broker reports none (a close submitted earlier is rejected
                      with "insufficient qty available")
  close_leg           idempotent close of one registry leg by client_order_id:
                      an earlier attempt that is still live is waited on, one
                      that filled is honoured, a dead one gets a new suffixed id;
                      returns the FILLED quantity, never "accepted"
  reconcile           registry vs broker with three explicit outcomes: broker
                      unreachable -> abort (nothing touched); broker holds less
                      -> reduce quantities oldest tranche first; broker holds
                      more / unknown symbol -> orphan report, never traded
  exposure            positions PLUS pending entry orders, per symbol and side
  entry_allowed       the no-debt guards on the cumulative book
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .broker import BrokerError, OrderState, LIVE, TERMINAL

ENTRY_PREFIX = "open_"


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------
def wait_for_terminal(broker, order_id: str, timeout_s: float = 30.0, poll_s: float = 1.0,
                      sleep: Callable[[float], None] = time.sleep) -> OrderState:
    """Poll until the order is terminal or the timeout passes; the returned
    state is whatever the broker last reported (possibly partially_filled)."""
    deadline = time.monotonic() + timeout_s
    last = broker.get_order(order_id)
    while not last.terminal and time.monotonic() < deadline:
        sleep(poll_s)
        last = broker.get_order(order_id)
    return last


def cancel_symbol_orders(broker, symbol: str, timeout_s: float = 15.0, poll_s: float = 1.0,
                         sleep: Callable[[float], None] = time.sleep, keep: Optional[set] = None) -> bool:
    """Cancel every live order in `symbol` except ids in `keep`; wait until
    the broker reports none of them open. Returns False on timeout."""
    keep = keep or set()
    for o in broker.open_orders(symbol):
        if o.id not in keep:
            broker.cancel(o.id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        live = [o for o in broker.open_orders(symbol) if o.id not in keep]
        if not live:
            return True
        sleep(poll_s)
    return False


@dataclass
class CloseResult:
    symbol: str
    requested: int
    filled: int
    status: str
    order_id: Optional[str]
    note: str = ""

    @property
    def complete(self) -> bool:
        return self.filled >= self.requested


def close_leg(broker, symbol: str, qty: int, side: str, client_order_id: str,
              timeout_s: float = 45.0, poll_s: float = 1.0,
              sleep: Callable[[float], None] = time.sleep, max_attempts: int = 3) -> CloseResult:
    """Close `qty` shares of a registry leg with a closing MARKET order.

    side is the closing order side ('sell' for a long leg, 'buy' for a short).
    Idempotent on client_order_id: attempt k uses `<cid>` then `<cid>_r2`,
    `<cid>_r3`; an existing live attempt is waited on instead of duplicated,
    a filled one is credited. Success means FILLED shares, not acceptance."""
    filled_total = 0
    last_status, last_id = "none", None
    for attempt in range(1, max_attempts + 1):
        cid = client_order_id if attempt == 1 else f"{client_order_id}_r{attempt}"
        remaining = qty - filled_total
        if remaining <= 0:
            break
        try:
            prior = broker.get_order_by_client_id(cid)
        except BrokerError as e:
            return CloseResult(symbol, qty, filled_total, "unknown", last_id, f"lookup failed: {e}")
        if prior is None:
            # nothing resting may hold the shares (bracket children)
            if not cancel_symbol_orders(broker, symbol, timeout_s=min(15.0, timeout_s), poll_s=poll_s, sleep=sleep):
                return CloseResult(symbol, qty, filled_total, "blocked", last_id, "open orders would not cancel")
            try:
                prior = broker.submit(symbol, remaining, side, kind="market", client_order_id=cid, tif="day")
            except BrokerError as e:
                return CloseResult(symbol, qty, filled_total, "submit_failed", last_id, str(e))
        state = wait_for_terminal(broker, prior.id, timeout_s=timeout_s, poll_s=poll_s, sleep=sleep)
        last_status, last_id = state.status, state.id
        filled_total += int(state.filled_qty)
        if state.status == "filled" or filled_total >= qty:
            return CloseResult(symbol, qty, min(filled_total, qty), "filled", state.id)
        if state.status in LIVE:
            # still working after the timeout (queued market order after the
            # close, halted stock): report what filled, keep the rest registered
            return CloseResult(symbol, qty, filled_total, state.status, state.id, "order still live at timeout")
        # terminal but not filled (rejected / canceled / expired): try a fresh id
    return CloseResult(symbol, qty, filled_total, last_status, last_id, "attempts exhausted")


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------
@dataclass
class ReconcileReport:
    aborted: bool = False
    reason: str = ""
    reduced: list = field(default_factory=list)     # (tranche_id, symbol, side, from_qty, to_qty)
    removed: list = field(default_factory=list)     # (tranche_id, symbol, side, qty)
    orphans: list = field(default_factory=list)     # (symbol, side, broker_qty, registry_qty)

    @property
    def changed(self) -> bool:
        return bool(self.reduced or self.removed)


def reconcile(broker, registry, dry_run: bool = False) -> ReconcileReport:
    """Converge the registry to broker truth WITHOUT ever destroying legs on
    a failed query. Broker holding less than the registry -> oldest tranche
    loses shares first (partial), not whole legs. Broker holding more, or a
    symbol the registry never opened -> orphan, reported and left alone."""
    rep = ReconcileReport()
    try:
        positions = broker.positions()
    except BrokerError as e:
        rep.aborted, rep.reason = True, f"broker positions unavailable: {e}"
        return rep
    held: dict[tuple[str, str], float] = {}
    for p in positions:
        held[(p.symbol, p.side)] = abs(float(p.qty))
    # registry totals per (symbol, side), oldest tranche first
    opens = sorted(registry.open_tranches(), key=lambda t: t.open_date)
    reg: dict[tuple[str, str], list] = {}
    for t in opens:
        for pos in t.longs:
            reg.setdefault((pos.symbol, "long"), []).append((t.id, pos))
        for pos in t.shorts:
            reg.setdefault((pos.symbol, "short"), []).append((t.id, pos))
    for key, legs in reg.items():
        reg_total = sum(int(p.qty) for _, p in legs)
        broker_qty = held.get(key, 0.0)
        if broker_qty >= reg_total:
            if broker_qty > reg_total:
                rep.orphans.append((key[0], key[1], broker_qty, reg_total))
            continue
        shortfall = reg_total - int(broker_qty)
        for tid, pos in legs:                        # oldest first
            if shortfall <= 0:
                break
            cut = min(int(pos.qty), shortfall)
            shortfall -= cut
            if cut >= int(pos.qty):
                rep.removed.append((tid, pos.symbol, key[1], int(pos.qty)))
                if not dry_run:
                    registry.remove_position(tid, pos.symbol, key[1], reason="reconciled_missing_at_broker")
            else:
                rep.reduced.append((tid, pos.symbol, key[1], int(pos.qty), int(pos.qty) - cut))
                if not dry_run:
                    registry.reduce_position(tid, pos.symbol, key[1], cut, reason="reconciled_partial_at_broker")
    for key, q in held.items():
        if key not in reg:
            rep.orphans.append((key[0], key[1], q, 0))
    if rep.changed and not dry_run:
        registry.save()
    return rep


# ---------------------------------------------------------------------------
# exposure and entry guards
# ---------------------------------------------------------------------------
@dataclass
class Exposure:
    gross: float = 0.0                 # |positions| + pending entry notional
    short: float = 0.0                 # short positions + pending short entries
    by_symbol_short: dict = field(default_factory=dict)
    by_symbol_gross: dict = field(default_factory=dict)
    pending_orders: int = 0


def exposure(broker, price_of: Optional[Callable[[str], Optional[float]]] = None) -> Exposure:
    """Positions PLUS resting entry orders (client ids prefixed 'open_'):
    an entry accepted but not yet filled is exposure the account has
    committed to. Raises BrokerError when the broker cannot be asked."""
    ex = Exposure()
    for p in broker.positions():
        mv = abs(float(p.market_value))
        ex.gross += mv
        ex.by_symbol_gross[p.symbol] = ex.by_symbol_gross.get(p.symbol, 0.0) + mv
        if p.qty < 0:
            ex.short += mv
            ex.by_symbol_short[p.symbol] = ex.by_symbol_short.get(p.symbol, 0.0) + mv
    for o in broker.open_orders():
        if not o.client_order_id.startswith(ENTRY_PREFIX):
            continue
        px = o.limit_price or (price_of(o.symbol) if price_of else None) or broker.latest_price(o.symbol) or 0.0
        notional = o.remaining * float(px)
        ex.pending_orders += 1
        ex.gross += notional
        ex.by_symbol_gross[o.symbol] = ex.by_symbol_gross.get(o.symbol, 0.0) + notional
        if o.side == "sell":
            ex.short += notional
            ex.by_symbol_short[o.symbol] = ex.by_symbol_short.get(o.symbol, 0.0) + notional
    return ex


def entry_allowed(ex: Exposure, symbol: str, side: str, notional: float, equity: float,
                  short_single_max_pct: float, short_gross_max_pct: float) -> tuple[bool, str]:
    """The no-debt guards on the CUMULATIVE book (positions + pending +
    this order). Returns (ok, reason)."""
    if ex.gross + notional > equity:
        return False, f"gross {ex.gross + notional:,.0f} > equity {equity:,.0f} -- no leverage, ever"
    if side == "short":
        single = ex.by_symbol_short.get(symbol, 0.0) + notional
        if single > equity * short_single_max_pct / 100.0:
            return False, f"single short {symbol} {single:,.0f} > {short_single_max_pct}% of equity (incl. existing)"
        if ex.short + notional > equity * short_gross_max_pct / 100.0:
            return False, f"aggregate short {ex.short + notional:,.0f} > {short_gross_max_pct}% of equity"
    return True, "ok"


def apply_entry(ex: Exposure, symbol: str, side: str, notional: float) -> None:
    """Book a just-placed entry into the snapshot so the next check in the
    same run sees it."""
    ex.gross += notional
    ex.by_symbol_gross[symbol] = ex.by_symbol_gross.get(symbol, 0.0) + notional
    if side == "short":
        ex.short += notional
        ex.by_symbol_short[symbol] = ex.by_symbol_short.get(symbol, 0.0) + notional


# ---------------------------------------------------------------------------
# circuit breaker actions
# ---------------------------------------------------------------------------
def flatten_all(broker, sleep: Callable[[float], None] = time.sleep, timeout_s: float = 60.0,
                poll_s: float = 1.0) -> dict:
    """Cancel EVERY open order (entries included), then close every position
    with market orders and wait for the fills. Returns a report; positions
    that did not close are listed so the caller can escalate."""
    report = {"canceled": 0, "closed": [], "unfilled": [], "errors": []}
    try:
        report["canceled"] = broker.cancel_all()
    except BrokerError as e:
        report["errors"].append(f"cancel_all: {e}")
    deadline = time.monotonic() + min(15.0, timeout_s)
    while time.monotonic() < deadline:
        try:
            if not broker.open_orders():
                break
        except BrokerError as e:
            report["errors"].append(f"open_orders: {e}"); break
        sleep(poll_s)
    try:
        positions = broker.positions()
    except BrokerError as e:
        report["errors"].append(f"positions: {e}")
        return report
    for p in positions:
        side = "sell" if p.qty > 0 else "buy"
        qty = int(abs(p.qty))
        res = close_leg(broker, p.symbol, qty, side, client_order_id=f"halt_{p.symbol}_{int(time.time())}",
                        timeout_s=timeout_s, poll_s=poll_s, sleep=sleep)
        (report["closed"] if res.complete else report["unfilled"]).append((p.symbol, qty, res.filled, res.status))
    return report
