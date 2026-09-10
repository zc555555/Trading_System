"""Broker interface for the execution layer (phase 1 of the 2026-09 review).

Two implementations share one contract:

  AlpacaBroker  thin adapter over alpaca-py's TradingClient; every failure
                raises BrokerError -- it NEVER returns an empty list for
                "I could not ask" (the old get_positions() did, and the
                reconciler read that as "the broker holds nothing").
  FakeBroker    in-memory broker for tests with fault injection: failed
                calls, delayed/partial fills, slow cancels, rejects; models
                bracket children (OCO legs that HOLD the position's shares
                exactly like Alpaca) and the exchange calendar.

Order states are lowercase strings as Alpaca reports them. Only the
TERMINAL set means the order is over; "accepted"/"new"/"pending_new"/
"held" mean the broker HAS the order, not that anything traded.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from .trading_calendar import is_trading_day, _to_date

TERMINAL = frozenset({"filled", "canceled", "cancelled", "expired", "rejected", "suspended", "done_for_day", "replaced"})
LIVE = frozenset({"accepted", "new", "pending_new", "held", "partially_filled", "accepted_for_bidding",
                  "pending_cancel", "pending_replace", "calculated", "stopped"})


class BrokerError(RuntimeError):
    """The broker could not be asked or refused the request (network,
    auth, 5xx, SDK exception). Callers must treat state as UNKNOWN."""


@dataclass
class OrderState:
    id: str
    client_order_id: str
    symbol: str
    side: str                 # 'buy' | 'sell'
    qty: float
    filled_qty: float
    status: str               # lowercase
    filled_avg_price: Optional[float] = None
    order_type: str = "market"
    order_class: str = "simple"
    limit_price: Optional[float] = None
    legs: list[str] = field(default_factory=list)      # child order ids of a bracket/OCO parent
    parent_id: Optional[str] = None                    # set on a child
    submitted_at: Optional[datetime] = None            # aware (UTC at Alpaca)
    children: list = field(default_factory=list)       # nested child OrderStates when the query was nested

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def remaining(self) -> float:
        return max(0.0, float(self.qty) - float(self.filled_qty))


@dataclass
class Position:
    symbol: str
    qty: float                # signed: negative = short
    avg_entry_price: float
    current_price: float
    market_value: float       # signed like qty
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"


def norm_status(raw) -> str:
    s = str(raw or "").lower()
    return s.split(".")[-1] if "." in s else s


# ---------------------------------------------------------------------------
# real broker
# ---------------------------------------------------------------------------
class AlpacaBroker:
    """Adapter over alpaca-py. Imports the SDK lazily so the rest of the
    package (and the tests) load without it."""

    def __init__(self, trading_client=None, data_client=None, paper: bool = True):
        if trading_client is None:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
            trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            paper = ALPACA_PAPER
        self.client = trading_client
        self.data = data_client
        self.paper = paper

    # -- queries ------------------------------------------------------------
    def account(self) -> dict:
        try:
            a = self.client.get_account()
            return {"cash": float(a.cash), "equity": float(a.equity), "last_equity": float(a.last_equity),
                    "buying_power": float(a.buying_power), "portfolio_value": float(a.portfolio_value)}
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_account failed: {e}") from e

    def positions(self) -> list[Position]:
        try:
            out = []
            for p in self.client.get_all_positions():
                out.append(Position(symbol=p.symbol, qty=float(p.qty), avg_entry_price=float(p.avg_entry_price),
                                    current_price=float(p.current_price), market_value=float(p.market_value),
                                    unrealized_pl=float(p.unrealized_pl), unrealized_plpc=float(p.unrealized_plpc)))
            return out
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_all_positions failed: {e}") from e

    def _to_state(self, o) -> OrderState:
        legs = getattr(o, "legs", None) or []
        return OrderState(id=str(o.id), client_order_id=str(o.client_order_id or ""), symbol=str(o.symbol),
                          side=norm_status(o.side), qty=float(o.qty or 0), filled_qty=float(o.filled_qty or 0),
                          status=norm_status(o.status),
                          filled_avg_price=float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None,
                          order_type=norm_status(getattr(o, "type", "market")),
                          order_class=norm_status(getattr(o, "order_class", "simple")),
                          limit_price=float(o.limit_price) if getattr(o, "limit_price", None) else None,
                          legs=[str(l.id) for l in legs],
                          children=[self._to_state(l) for l in legs],
                          parent_id=str(getattr(o, "parent_id", None) or "") or None,
                          submitted_at=getattr(o, "submitted_at", None) or getattr(o, "created_at", None))

    def open_orders(self, symbol: Optional[str] = None) -> list[OrderState]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol] if symbol else None, nested=True)
            return [self._to_state(o) for o in self.client.get_orders(req)]
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_orders failed: {e}") from e

    def get_order(self, order_id: str) -> OrderState:
        """Nested: a bracket parent comes back with its child ids in .legs."""
        try:
            from alpaca.trading.requests import GetOrderByIdRequest
            return self._to_state(self.client.get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True)))
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_order_by_id failed: {e}") from e

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderState]:
        try:
            return self._to_state(self.client.get_order_by_client_id(client_order_id))
        except Exception as e:                       # noqa: BLE001
            msg = str(e).lower()
            if "not found" in msg or "404" in msg:
                return None
            raise BrokerError(f"get_order_by_client_id failed: {e}") from e

    def latest_price(self, symbol: str) -> Optional[float]:
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            t = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            px = float(t[symbol].price)
            return px if px > 0 else None
        except Exception:                            # noqa: BLE001
            return None

    def orders_since(self, after: datetime) -> list[OrderState]:
        """Every order (any status, nested children) submitted after `after`.
        The reconciler converges the registry from these -- our own client
        ids are the only ground truth about which tranche owns what."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            out, cursor = [], after
            for _ in range(20):                                  # 500 per page
                page = self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True,
                                                               limit=500, after=cursor, direction="asc"))
                states = [self._to_state(o) for o in page]
                out.extend(states)
                if len(states) < 500:
                    break
                cursor = max(s.submitted_at for s in states if s.submitted_at)
            return out
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_orders(all) failed: {e}") from e

    def calendar(self, start: date, end: date) -> list[date]:
        """Trading sessions in [start, end] according to the exchange."""
        try:
            from alpaca.trading.requests import GetCalendarRequest
            days = self.client.get_calendar(GetCalendarRequest(start=start, end=end))
            return [_to_date(c.date) for c in days]
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"get_calendar failed: {e}") from e

    # -- mutations ----------------------------------------------------------
    def submit(self, symbol: str, qty: int, side: str, kind: str = "market", limit_price: Optional[float] = None,
               bracket: Optional[tuple[float, float]] = None, client_order_id: Optional[str] = None,
               tif: str = "day") -> OrderState:
        try:
            from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest, StopLossRequest,
                                                 TakeProfitRequest)
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
            kw = dict(symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                      time_in_force=TimeInForce.GTC if tif == "gtc" else TimeInForce.DAY,
                      client_order_id=client_order_id)
            if bracket is not None:
                stop, take = bracket
                kw.update(order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=round(stop, 2)),
                          take_profit=TakeProfitRequest(limit_price=round(take, 2)))
            req = LimitOrderRequest(limit_price=round(float(limit_price), 2), **kw) if kind == "limit" else MarketOrderRequest(**kw)
            return self._to_state(self.client.submit_order(req))
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"submit_order failed for {symbol}: {e}") from e

    def cancel(self, order_id: str) -> None:
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"cancel failed for {order_id}: {e}") from e

    def cancel_all(self) -> int:
        try:
            res = self.client.cancel_orders()
            return len(res) if res is not None else 0
        except Exception as e:                       # noqa: BLE001
            raise BrokerError(f"cancel_orders failed: {e}") from e


# ---------------------------------------------------------------------------
# fake broker for tests
# ---------------------------------------------------------------------------
class FakeBroker:
    """In-memory broker with fault injection.

    fail_next[method] = n      the next n calls of that method raise BrokerError
    fill_delay_polls           market orders stay 'accepted' for this many polls
    partial_fill[symbol] = f   a market order in symbol fills only fraction f (rest stays live)
    cancel_delay_polls         a cancel takes this many polls to become terminal
    reject[symbol] = reason    submissions in symbol are rejected
    holidays                   extra non-trading dates for calendar()

    Bracket entries create two child orders (stop + take-profit, OCO) that
    stay live and HOLD the entry's shares: a closing order for more shares
    than are free is rejected ("insufficient qty available"), which is what
    Alpaca does and why the close path must cancel the right children.
    """

    def __init__(self, equity: float = 100_000.0, last_equity: Optional[float] = None, cash: Optional[float] = None,
                 prices: Optional[dict] = None, holidays: Optional[set] = None):
        self.equity = float(equity)
        self.last_equity = float(last_equity if last_equity is not None else equity)
        self.cash = float(cash if cash is not None else equity)
        self.prices: dict[str, float] = dict(prices or {})
        self.pos: dict[str, float] = {}              # signed qty
        self.avg: dict[str, float] = {}
        self.orders: dict[str, OrderState] = {}
        self._pending_polls: dict[str, int] = {}
        self._cancel_polls: dict[str, int] = {}
        self._ids = itertools.count(1)
        self.fail_next: dict[str, int] = {}
        self.fill_delay_polls = 0
        self.partial_fill: dict[str, float] = {}
        self.cancel_delay_polls = 0
        self.reject: dict[str, str] = {}
        self.holidays: set = set(holidays or ())
        self.clock: Optional[datetime] = None        # submitted_at for new orders (None = now)
        self.calls: list[tuple] = []

    # -- fault helpers ------------------------------------------------------
    def _maybe_fail(self, method: str):
        n = self.fail_next.get(method, 0)
        if n > 0:
            self.fail_next[method] = n - 1
            raise BrokerError(f"injected failure: {method}")

    def _advance(self):
        """One poll: pending fills and cancels progress."""
        for oid, o in list(self.orders.items()):
            if oid in self._cancel_polls:
                self._cancel_polls[oid] -= 1
                if self._cancel_polls[oid] <= 0:
                    del self._cancel_polls[oid]
                    o.status = "canceled"
                continue
            if o.status in TERMINAL or o.order_type != "market":
                continue
            if self._pending_polls.get(oid, 0) > 0:
                self._pending_polls[oid] -= 1
                continue
            self._fill(o)

    def _fill(self, o: OrderState):
        frac = self.partial_fill.get(o.symbol, 1.0)
        target = float(o.qty) if frac >= 1.0 else float(int(o.qty * frac))
        delta = target - o.filled_qty
        if delta <= 0:
            o.status = "partially_filled" if 0 < o.filled_qty < o.qty else o.status
            return
        px = self.prices.get(o.symbol, 100.0)
        signed = delta if o.side == "buy" else -delta
        old = self.pos.get(o.symbol, 0.0)
        new = old + signed
        if new != 0 and (old == 0 or (old > 0) == (new > 0)) and abs(new) > abs(old):
            self.avg[o.symbol] = (abs(old) * self.avg.get(o.symbol, px) + abs(signed) * px) / abs(new)
        elif new == 0:
            self.avg.pop(o.symbol, None)
        self.pos[o.symbol] = new
        if new == 0:
            del self.pos[o.symbol]
        self.cash -= signed * px
        o.filled_qty = target
        o.filled_avg_price = px
        o.status = "filled" if o.filled_qty >= o.qty else "partially_filled"
        if o.status == "filled" and o.legs:
            for lid in o.legs:                       # children become working orders once the parent fills
                child = self.orders.get(lid)
                if child is not None and child.status == "held":
                    child.status = "new"

    def held_qty(self, symbol: str, side: str) -> float:
        """Shares of `symbol` tied up by live non-market orders on `side`
        (bracket children, resting stops/limits)."""
        held, parents_seen = 0.0, set()
        for o in self.orders.values():
            if o.symbol != symbol or o.side != side or o.status not in LIVE or o.order_type == "market":
                continue
            if o.parent_id:                          # an OCO pair holds its qty once, not twice
                if o.parent_id in parents_seen:
                    continue
                parents_seen.add(o.parent_id)
            held += o.remaining
        return held

    # -- queries ------------------------------------------------------------
    def account(self) -> dict:
        self._maybe_fail("account")
        return {"cash": self.cash, "equity": self.equity, "last_equity": self.last_equity,
                "buying_power": self.equity, "portfolio_value": self.equity}

    def positions(self) -> list[Position]:
        self._maybe_fail("positions")
        out = []
        for s, q in self.pos.items():
            px = self.prices.get(s, 100.0)
            out.append(Position(symbol=s, qty=q, avg_entry_price=self.avg.get(s, px), current_price=px,
                                market_value=q * px, unrealized_pl=(px - self.avg.get(s, px)) * q,
                                unrealized_plpc=(px / self.avg.get(s, px) - 1.0) * (1 if q > 0 else -1)))
        return out

    def open_orders(self, symbol: Optional[str] = None) -> list[OrderState]:
        self._maybe_fail("open_orders")
        self._advance()
        return [o for o in self.orders.values() if o.status in LIVE and (symbol is None or o.symbol == symbol)]

    def get_order(self, order_id: str) -> OrderState:
        self._maybe_fail("get_order")
        self._advance()
        if order_id not in self.orders:
            raise BrokerError(f"order {order_id} not found")
        return self.orders[order_id]

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderState]:
        self._maybe_fail("get_order_by_client_id")
        self._advance()
        for o in self.orders.values():
            if o.client_order_id == client_order_id:
                return o
        return None

    def latest_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol)

    def calendar(self, start: date, end: date) -> list[date]:
        self._maybe_fail("calendar")
        out, cur = [], _to_date(start)
        while cur <= _to_date(end):
            if is_trading_day(cur) and cur not in self.holidays:
                out.append(cur)
            cur += timedelta(days=1)
        return out

    # -- mutations ----------------------------------------------------------
    def submit(self, symbol: str, qty: int, side: str, kind: str = "market", limit_price: Optional[float] = None,
               bracket: Optional[tuple[float, float]] = None, client_order_id: Optional[str] = None,
               tif: str = "day") -> OrderState:
        self._maybe_fail("submit")
        self.calls.append(("submit", symbol, qty, side, kind, client_order_id))
        if client_order_id and any(o.client_order_id == client_order_id for o in self.orders.values()):
            raise BrokerError(f"client_order_id {client_order_id} already used")
        oid = f"fake-{next(self._ids)}"
        o = OrderState(id=oid, client_order_id=client_order_id or oid, symbol=symbol, side=side, qty=float(qty),
                       filled_qty=0.0, status="accepted", order_type=kind,
                       order_class="bracket" if bracket else "simple", limit_price=limit_price,
                       submitted_at=self.clock or datetime.now(timezone.utc))
        if symbol in self.reject:
            o.status = "rejected"
        else:
            pos = self.pos.get(symbol, 0.0)
            reduces = (pos > 0 and side == "sell") or (pos < 0 and side == "buy")
            if reduces and bracket is None:
                free = abs(pos) - self.held_qty(symbol, side)
                if float(qty) > free + 1e-9:
                    o.status = "rejected"
                    self.calls.append(("reject", symbol, f"insufficient qty available for order (requested: {qty}, available: {free:.0f})"))
        self.orders[oid] = o
        if o.status != "rejected" and bracket is not None:
            stop, take = bracket
            exit_side = "sell" if side == "buy" else "buy"
            for kind_, px in (("stop", stop), ("limit", take)):
                cid = f"fake-{next(self._ids)}"
                child = OrderState(id=cid, client_order_id=cid, symbol=symbol, side=exit_side, qty=float(qty),
                                   filled_qty=0.0, status="held", order_type=kind_, order_class="bracket",
                                   limit_price=px, parent_id=oid)
                self.orders[cid] = child
                o.legs.append(cid)
                o.children.append(child)
        if self.fill_delay_polls and o.status != "rejected":
            self._pending_polls[oid] = self.fill_delay_polls
        return o

    def orders_since(self, after: datetime) -> list[OrderState]:
        self._maybe_fail("orders_since")
        self._advance()
        out = []
        for o in self.orders.values():
            sub = o.submitted_at
            if sub is not None and sub.tzinfo is None:
                sub = sub.replace(tzinfo=timezone.utc)
            if sub is None or after is None or sub >= after:
                out.append(o)
        return out

    def trigger_exit(self, parent_id: str, which: str = "stop") -> OrderState:
        """Fill one bracket child of a filled entry (stop or take): the
        position shrinks by the child's qty, its OCO sibling is cancelled."""
        parent = self.orders[parent_id]
        kind = "stop" if which == "stop" else "limit"
        child = next(c for c in parent.children if c.order_type == kind)
        px = float(child.limit_price or self.prices.get(child.symbol, 100.0))
        signed = child.qty if child.side == "buy" else -child.qty
        new = self.pos.get(child.symbol, 0.0) + signed
        if abs(new) < 1e-9:
            self.pos.pop(child.symbol, None); self.avg.pop(child.symbol, None)
        else:
            self.pos[child.symbol] = new
        self.cash -= signed * px
        child.filled_qty, child.filled_avg_price, child.status = child.qty, px, "filled"
        for c in parent.children:
            if c is not child and c.status in LIVE:
                c.status = "canceled"
        return child

    def _cancel_one(self, o: OrderState) -> None:
        if o.status in TERMINAL:
            return
        if self.cancel_delay_polls:
            self._cancel_polls[o.id] = self.cancel_delay_polls
            o.status = "pending_cancel"
        else:
            o.status = "canceled"

    def cancel(self, order_id: str) -> None:
        self._maybe_fail("cancel")
        self.calls.append(("cancel", order_id))
        o = self.orders.get(order_id)
        if o is None:
            raise BrokerError(f"order {order_id} not found")
        if o.status in TERMINAL:
            return
        self._cancel_one(o)
        # OCO: cancelling one bracket child cancels its sibling; cancelling a
        # live parent cancels its children
        siblings = []
        if o.parent_id and o.parent_id in self.orders:
            siblings = [self.orders[l] for l in self.orders[o.parent_id].legs if l != o.id and l in self.orders]
        elif o.legs:
            siblings = [self.orders[l] for l in o.legs if l in self.orders]
        for s in siblings:
            self._cancel_one(s)

    def cancel_all(self) -> int:
        self._maybe_fail("cancel_all")
        n = 0
        for oid, o in list(self.orders.items()):
            if o.status in LIVE:
                self.cancel(oid); n += 1
        return n
