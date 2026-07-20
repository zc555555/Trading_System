"""Tranche registry for staggered multi-day trading.

A *tranche* is a batch of positions opened on a single trading day and held
for ``hold_days`` trading days, then closed. Running N tranches in parallel
spreads exposure across N entry dates: on day N+1 the oldest tranche closes
and a fresh one opens, so portfolio composition rolls smoothly rather than
churning 100% at once.

This module owns the on-disk JSON registry and provides three operations:

    registry.add_tranche(tranche_id, positions, open_date, hold_days, ...)
    registry.tranches_due_to_close(today)
    registry.mark_closed(tranche_id, realized_pnl, reason)

The orchestrator (``run_staggered_trading.py``) is the only intended caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .trading_calendar import (
    DateLike,
    _to_date,
    add_trading_days,
)


@dataclass
class TranchePosition:
    """One leg of a tranche -- a single long or short position in one symbol."""
    symbol: str
    side: str                  # 'long' or 'short'
    qty: int
    entry_price: float
    entry_amount_usd: float
    client_order_id: str
    prediction: float = 0.0     # raw signal at entry, for audit
    sentiment: float = 0.0
    # W2-C (ATR-adaptive stops): absolute dollar prices computed at entry.
    # Monitor reads these directly and triggers when current_price crosses them.
    # None = no per-position stop set; monitor falls back to fixed-% from config.
    stop_price: Optional[float] = None
    take_price: Optional[float] = None
    # Diagnostics for the audit log
    atr_at_entry: Optional[float] = None
    stop_basis: Optional[str] = None      # 'atr' | 'fixed_pct_fallback'


@dataclass
class Tranche:
    id: str
    open_date: str             # YYYY-MM-DD
    scheduled_close_date: str  # YYYY-MM-DD
    status: str                # 'open' | 'closed' | 'partially_closed'
    longs: List[TranchePosition] = field(default_factory=list)
    shorts: List[TranchePosition] = field(default_factory=list)
    signal_file: Optional[str] = None
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Tranche":
        d = dict(d)
        d["longs"] = [TranchePosition(**p) for p in d.get("longs", [])]
        d["shorts"] = [TranchePosition(**p) for p in d.get("shorts", [])]
        return cls(**d)


class TrancheRegistry:
    """Append-only registry of tranches with safe atomic writes."""

    VERSION = 1
    SCHEMA_HELP = (
        "Each tranche is a batch of positions opened on the same trading day "
        "and held until scheduled_close_date. The orchestrator closes them in "
        "FIFO order."
    )

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._payload: dict = self._load()

    # ----- IO -----
    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "version": self.VERSION,
                "schema_help": self.SCHEMA_HELP,
                "created_at": datetime.now().isoformat(),
                "tranches": [],
            }
        with open(self.path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != self.VERSION:
            raise RuntimeError(
                f"Registry version mismatch in {self.path}: "
                f"got {payload.get('version')}, expected {self.VERSION}"
            )
        return payload

    def save(self) -> None:
        self._payload["last_updated"] = datetime.now().isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._payload, f, indent=2)
        tmp.replace(self.path)

    # ----- introspection -----
    @property
    def tranches(self) -> List[Tranche]:
        return [Tranche.from_dict(t) for t in self._payload["tranches"]]

    def open_tranches(self) -> List[Tranche]:
        return [t for t in self.tranches if t.status == "open"]

    def get(self, tranche_id: str) -> Optional[Tranche]:
        for t in self.tranches:
            if t.id == tranche_id:
                return t
        return None

    def tranches_due_to_close(self, as_of: DateLike) -> List[Tranche]:
        """All open tranches with scheduled_close_date <= as_of."""
        as_of_d = _to_date(as_of)
        out = []
        for t in self.open_tranches():
            if _to_date(t.scheduled_close_date) <= as_of_d:
                out.append(t)
        return out

    # ----- mutations -----
    def add_tranche(
        self,
        tranche_id: str,
        open_date: DateLike,
        hold_days: int,
        longs: Sequence[TranchePosition] = (),
        shorts: Sequence[TranchePosition] = (),
        signal_file: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Tranche:
        if self.get(tranche_id) is not None:
            raise ValueError(f"Tranche {tranche_id} already exists")
        scheduled = add_trading_days(open_date, hold_days)
        t = Tranche(
            id=tranche_id,
            open_date=_to_date(open_date).isoformat(),
            scheduled_close_date=scheduled.isoformat(),
            status="open",
            longs=list(longs),
            shorts=list(shorts),
            signal_file=signal_file,
            notes=notes,
        )
        self._payload["tranches"].append(t.to_dict())
        return t

    def mark_closed(
        self,
        tranche_id: str,
        realized_pnl: Optional[float],
        reason: str,
        closed_at: Optional[str] = None,
    ) -> None:
        for rec in self._payload["tranches"]:
            if rec["id"] == tranche_id:
                rec["status"] = "closed"
                rec["realized_pnl"] = realized_pnl
                rec["close_reason"] = reason
                rec["closed_at"] = closed_at or datetime.now().isoformat()
                return
        raise KeyError(f"Tranche {tranche_id} not found")

    def add_position(self, tranche_id: str, pos: TranchePosition) -> None:
        """Append one position to an existing tranche.

        Used by the orchestrator to record each leg IMMEDIATELY after its
        order is accepted (followed by save()), so a crash mid-tranche never
        leaves live broker positions missing from the registry.
        """
        for rec in self._payload["tranches"]:
            if rec["id"] == tranche_id:
                key = "longs" if pos.side == "long" else "shorts"
                rec[key].append(asdict(pos))
                return
        raise KeyError(f"Tranche {tranche_id} not found")

    def remove_position(self, tranche_id: str, symbol: str, side: str,
                        reason: str = "all_positions_stopped_out") -> bool:
        """Remove a single position (e.g. after a stop-loss or a confirmed
        scheduled close). If the tranche becomes empty it is marked closed
        with ``reason``.

        Returns True if a position was removed, False if nothing matched.
        """
        for rec in self._payload["tranches"]:
            if rec["id"] != tranche_id:
                continue
            key = "longs" if side == "long" else "shorts"
            before = len(rec[key])
            rec[key] = [p for p in rec[key] if p["symbol"] != symbol]
            removed = before > len(rec[key])
            if removed and not rec["longs"] and not rec["shorts"]:
                rec["status"] = "closed"
                rec["close_reason"] = reason
                rec["closed_at"] = datetime.now().isoformat()
            return removed
        return False

    # ----- aggregates -----
    def equity_in_open_tranches(self) -> float:
        total = 0.0
        for t in self.open_tranches():
            for p in t.longs + t.shorts:
                total += p.entry_amount_usd
        return total

    def n_open_tranches(self) -> int:
        return len(self.open_tranches())

    def summary(self) -> dict:
        opens = self.open_tranches()
        return {
            "n_tranches_total": len(self.tranches),
            "n_open": len(opens),
            "n_open_long_positions": sum(len(t.longs) for t in opens),
            "n_open_short_positions": sum(len(t.shorts) for t in opens),
            "capital_deployed_usd": self.equity_in_open_tranches(),
        }
