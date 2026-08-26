"""
Staggered multi-day trading orchestrator (W2-B, 2026-05).

Each invocation runs ONE daily cycle:
    1. Close any tranches whose scheduled_close_date <= today.
    2. Open a NEW tranche with capital_per_tranche_pct of account equity,
       sized by today's signals (long for BUY, short for SELL).
    3. Persist the updated tranche registry.

Designed to replace ``run_auto_trading.py`` once W2-B is validated. The old
1-day-hold script is kept untouched so you can roll back by editing
config_trading.STRATEGY back to "legacy".

CLI:
    python run_staggered_trading.py
    python run_staggered_trading.py --dry-run     # don't send orders
    python run_staggered_trading.py --close-only  # close due tranches, skip new
    python run_staggered_trading.py --open-only   # open new, skip closes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import config_trading
from trading.tranche_registry import (
    TrancheRegistry,
    TranchePosition,
)
from trading.risk_levels import compute_risk_levels
from trading.atr_loader import atr_summary

# Local imports for the existing trading infra
sys.path.insert(0, str(Path(__file__).parent))
from alpaca_trader import AlpacaAutoTrader  # noqa: E402


ROOT = Path(__file__).parent


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def exec_arm_for(symbol: str, date_str: str) -> str:
    """Deterministic A/B assignment: reproducible, balanced-in-expectation,
    uncorrelated with the signal (hash of symbol+date)."""
    import hashlib
    h = hashlib.sha256(f"{symbol}|{date_str}".encode()).digest()
    return "limit" if h[0] % 2 else "market"


def _today_id_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# CLOSE LEG
# ---------------------------------------------------------------------------
def close_due_tranches(trader: AlpacaAutoTrader, registry: TrancheRegistry,
                       dry_run: bool) -> int:
    """Close all tranches whose scheduled_close_date <= today."""
    due = registry.tranches_due_to_close(_today_str())
    if not due:
        print("[close] No tranches due today.")
        return 0

    print(f"\n{'=' * 70}\nCLOSING {len(due)} DUE TRANCHE(S)\n{'=' * 70}")
    n_orders = 0
    for tranche in due:
        n_pos = len(tranche.longs) + len(tranche.shorts)
        print(f"\n[close] {tranche.id}  open={tranche.open_date}  "
              f"sched_close={tranche.scheduled_close_date}  positions={n_pos}")

        # (position, close-order side, book side) — long -> sell, short -> buy to cover
        legs = ([(pos, "sell", "long") for pos in tranche.longs]
                + [(pos, "buy", "short") for pos in tranche.shorts])

        realized_total = 0.0
        n_failed = 0
        for pos, order_side, side in legs:
            ok = _close_position(trader, pos.symbol, pos.qty, order_side,
                                 client_order_id=f"close_{tranche.id}_{pos.symbol}",
                                 dry_run=dry_run)
            if not ok:
                # Leg stays in the registry; the tranche remains 'open' and
                # past-due, so the next run retries it. Never mark a position
                # closed that the broker may still hold.
                n_failed += 1
                print(f"  [FAIL] close {order_side.upper()} {pos.qty} {pos.symbol} "
                      f"-- position kept in registry for retry")
                continue

            n_orders += 1
            # P&L estimate based on current price; settlement may differ slightly
            cur = trader.get_current_price(pos.symbol)
            if cur:
                if side == "long":
                    pnl = (cur - pos.entry_price) * pos.qty
                else:
                    pnl = (pos.entry_price - cur) * pos.qty
                realized_total += pnl
                tag = "[SELL  long]" if side == "long" else "[BUY  short]"
                print(f"  {tag}  {pos.symbol:<6} qty={pos.qty:>4}  "
                      f"entry=${pos.entry_price:>7.2f}  cur=${cur:>7.2f}  "
                      f"P&L=${pnl:+.2f}")

            if not dry_run:
                # Persist each confirmed close immediately so a crash between
                # legs can't desync registry vs broker.
                registry.remove_position(tranche.id, pos.symbol, side,
                                         reason="scheduled_close")
                registry.save()

        if not dry_run:
            if n_failed == 0:
                # remove_position() emptied the tranche and marked it closed;
                # record the realized P&L estimate on top.
                registry.mark_closed(tranche.id, realized_total,
                                     reason="scheduled_close")
                registry.save()
            else:
                print(f"  [WARN] {tranche.id}: {n_failed} leg(s) failed to close; "
                      f"tranche stays open and will be retried next run")

    return n_orders


def _close_position(trader: AlpacaAutoTrader, symbol: str, qty: int, side: str,
                    client_order_id: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [DRY-RUN] would close: {side.upper()} {qty} {symbol}")
        return True
    # Resting bracket legs (stop/take) hold the position's qty; cancel them
    # first or the closing market order is rejected. cancel_open_orders now
    # waits for the broker to process the cancels; one retry after a short
    # pause covers a slow cancel (2026-08-26 race: 6 closes rejected with
    # "insufficient qty available ... held_for_orders").
    trader.cancel_open_orders(symbol)
    if trader.place_market_order(symbol, qty, side):
        return True
    time.sleep(3)
    trader.wait_until_no_open_orders(symbol)
    print(f"  [retry] close {side.upper()} {qty} {symbol}")
    return trader.place_market_order(symbol, qty, side)


def reconcile_registry_with_broker(trader: AlpacaAutoTrader,
                                   registry: TrancheRegistry,
                                   dry_run: bool) -> int:
    """Drop registry legs the broker no longer holds.

    A broker-side bracket leg may have fired overnight (stop or take), or a
    position may have been closed manually / by the monitor. The registry
    must converge to broker truth, otherwise the close leg retries dead
    positions forever. Compares per (symbol, side) share totals and removes
    surplus registry legs oldest-tranche-first.
    """
    positions = {p['symbol']: p for p in trader.get_positions()}
    removed = 0
    for tranche in registry.open_tranches():
        for pos in list(tranche.longs) + list(tranche.shorts):
            broker = positions.get(pos.symbol)
            broker_qty = float(broker['qty']) if broker else 0.0
            held = broker_qty if pos.side == "long" else -broker_qty
            # Sum of registry qty for this symbol+side across open tranches
            reg_qty = sum(
                p.qty for t in registry.open_tranches()
                for p in (t.longs if pos.side == "long" else t.shorts)
                if p.symbol == pos.symbol)
            if held >= reg_qty:
                continue
            print(f"  [reconcile] {pos.symbol} {pos.side}: registry={reg_qty} "
                  f"broker={held:.0f} -- removing {tranche.id} leg "
                  f"(bracket fired or external close)")
            if not dry_run:
                registry.remove_position(tranche.id, pos.symbol, pos.side,
                                         reason="reconciled_missing_at_broker")
                registry.save()
            removed += 1
    if not removed:
        print("  [reconcile] registry matches broker")
    return removed


# ---------------------------------------------------------------------------
# OPEN LEG
# ---------------------------------------------------------------------------
def load_today_signals() -> dict | None:
    """Load today's signal file from research/artifacts/.

    Falls back to yesterday's file if today's doesn't exist (so a missed
    overnight signal-gen doesn't kill the day's trades).
    """
    artifacts = ROOT / "research" / "artifacts"
    for offset in (0, -1):
        d = datetime.now().date()
        from datetime import timedelta
        target = d + timedelta(days=offset)
        path = artifacts / f"signals_multi_factor_{target.strftime('%Y%m%d')}.json"
        if path.exists():
            print(f"[signals] Using {path.name}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    print("[signals] No signal file found for today or yesterday -- skipping new tranche.")
    return None


def open_new_tranche(trader: AlpacaAutoTrader, registry: TrancheRegistry,
                     signals: dict, dry_run: bool) -> str | None:
    """Build today's tranche from signals; place orders; register."""
    if not signals.get("should_trade"):
        print(f"[open] should_trade=False (n_stocks={signals.get('n_stocks', 0)})"
              f" -- no tranche opened today.")
        return None

    account = trader.get_account_info()
    if not account:
        print("[open] Cannot fetch account; aborting open-leg.")
        return None

    equity = float(account["equity"])
    tranche_pct = config_trading.derived_capital_per_tranche_pct()
    tranche_capital = equity * (tranche_pct / 100.0)

    per_stock_max = (config_trading.PER_STOCK_MAX_PCT / 100.0) * tranche_capital

    tranche_id = f"tranche_{_today_id_stamp()}"
    if registry.get(tranche_id) is not None:
        print(f"[open] {tranche_id} already exists -- already ran today. Skipping.")
        return None

    # -------- no-debt principle: exposure snapshot for the hard guards ----
    # (config_trading header block, 2026-08-11)
    positions = trader.get_positions()
    gross_now = sum(abs(p['market_value']) for p in positions)
    short_now = sum(abs(p['market_value']) for p in positions
                    if float(p['qty']) < 0)
    placed_notional = 0.0
    placed_short = 0.0

    print(f"\n{'=' * 70}\nOPENING NEW TRANCHE {tranche_id}\n{'=' * 70}")
    print(f"  [no-debt] gross now ${gross_now:,.0f} "
          f"(short ${short_now:,.0f}) vs equity ${equity:,.0f}")
    print(f"  account equity:     ${equity:>12,.2f}")
    print(f"  tranche capital:    ${tranche_capital:>12,.2f}  "
          f"({tranche_pct:.1f}% of equity)")
    print(f"  per-stock max:      ${per_stock_max:>12,.2f}  "
          f"({config_trading.PER_STOCK_MAX_PCT}% of tranche)")
    print(f"  hold_days:          {config_trading.HOLD_DAYS}")
    print(f"  shorts allowed:     {config_trading.ALLOW_SHORTS}")
    if config_trading.USE_ATR_STOPS:
        print(f"  risk model:         ATR-adaptive (stop={config_trading.STOP_ATR_MULTIPLE}x,"
              f" take={config_trading.TAKE_ATR_MULTIPLE}x ATR_14)")
        print(f"  [atr]               {atr_summary()}")
    else:
        print(f"  risk model:         fixed {config_trading.STOP_LOSS_PCT*100:.1f}%/"
              f"{config_trading.TAKE_PROFIT_PCT*100:.1f}%")

    longs: list[TranchePosition] = []
    shorts: list[TranchePosition] = []
    tranche_created = False

    for stock in signals["stocks"]:
        symbol = stock["symbol"]
        pred = float(stock["prediction"])
        pos_pct = float(stock["position_pct"]) / 100.0
        side = "long" if pred > 0 else "short"

        if side == "short":
            if config_trading.LONG_ONLY_FILTER:
                print(f"  [SKIP] {symbol}: SELL signal but LONG_ONLY_FILTER=True")
                continue
            if not config_trading.ALLOW_SHORTS:
                print(f"  [SKIP] {symbol}: SELL signal but ALLOW_SHORTS=False")
                continue

        raw_alloc = tranche_capital * pos_pct
        alloc = min(raw_alloc, per_stock_max)

        cur_price = trader.get_current_price(symbol)
        if cur_price is None or cur_price <= 0:
            print(f"  [SKIP] {symbol}: no price")
            continue

        qty = int(alloc // cur_price)
        if qty < 1:
            print(f"  [SKIP] {symbol}: qty<1 (alloc=${alloc:.2f} / price=${cur_price:.2f})")
            continue

        # ---- no-debt principle hard guards (these are NOT buying-power
        # checks: margin buying power allows leverage, equity does not) ----
        order_notional = qty * cur_price
        if gross_now + placed_notional + order_notional > equity:
            print(f"  [BLOCK] {symbol}: would exceed 100% equity exposure "
                  f"(gross ${gross_now + placed_notional:,.0f} + "
                  f"${order_notional:,.0f} > ${equity:,.0f}) -- no leverage, ever")
            continue
        if side == "short":
            if order_notional > equity * config_trading.SHORT_SINGLE_MAX_PCT / 100:
                print(f"  [BLOCK] {symbol}: single short cap "
                      f"({config_trading.SHORT_SINGLE_MAX_PCT}% equity)")
                continue
            if short_now + placed_short + order_notional > \
                    equity * config_trading.SHORT_GROSS_MAX_PCT / 100:
                print(f"  [BLOCK] {symbol}: aggregate short cap "
                      f"({config_trading.SHORT_GROSS_MAX_PCT}% equity)")
                continue

        # P2: never submit more notional than the account can carry.
        if not dry_run:
            fresh = trader.get_account_info()
            buying_power = float(fresh["buying_power"]) if fresh else 0.0
            if order_notional > buying_power:
                print(f"  [SKIP] {symbol}: insufficient buying power "
                      f"(need ${order_notional:,.0f}, have ${buying_power:,.0f})")
                continue

        client_id = f"open_{tranche_id}_{symbol}"
        order_side = "buy" if side == "long" else "sell"

        # Compute per-position stop & take prices BEFORE the order so we can
        # log them alongside the entry. Risk levels are persisted with the
        # tranche so the monitor reads exact prices, not recomputed thresholds.
        risk = compute_risk_levels(symbol, float(cur_price), side)
        arm = (exec_arm_for(symbol, _today_str())
               if config_trading.EXECUTION_AB_TEST else "market")

        if dry_run:
            ok = True
            print(f"  [DRY-RUN] {order_side.upper():<4} {side:<5} "
                  f"{symbol:<6} qty={qty:>4} @ ${cur_price:>7.2f} "
                  f"= ${qty*cur_price:>9,.2f}  "
                  f"stop=${risk.stop_price:>7.2f}({risk.stop_pct*100:.1f}%) "
                  f"take=${risk.take_price:>7.2f}({risk.take_pct*100:.1f}%) "
                  f"[{risk.basis}, {arm}]")
        else:
            # P2: bracket order -- stop & take live at the BROKER (GTC), so
            # multi-day positions stay protected overnight and across crashes.
            # Execution A/B (2026-08): arm B posts a LIMIT at the arrival
            # price instead of paying the open; monitor converts non-fills.
            if arm == "limit":
                ok = trader.place_limit_bracket_order(
                    symbol, qty, order_side, limit_price=float(cur_price),
                    stop_price=risk.stop_price, take_price=risk.take_price,
                    client_order_id=client_id)
            else:
                ok = trader.place_bracket_order(
                    symbol, qty, order_side,
                    stop_price=risk.stop_price, take_price=risk.take_price,
                    client_order_id=client_id)
            if ok:
                print(f"  [OK]      {order_side.upper():<4} {side:<5} "
                      f"{symbol:<6} qty={qty:>4} @ ${cur_price:>7.2f} "
                      f"= ${qty*cur_price:>9,.2f}  "
                      f"stop=${risk.stop_price:>7.2f}({risk.stop_pct*100:.1f}%) "
                      f"take=${risk.take_price:>7.2f}({risk.take_pct*100:.1f}%) "
                      f"[{risk.basis}, {arm}@broker]")
            else:
                print(f"  [FAIL]    {order_side.upper():<4} {side:<5} {symbol}")
                continue

        pos = TranchePosition(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=float(cur_price),
            entry_amount_usd=float(qty * cur_price),
            client_order_id=client_id,
            prediction=pred,
            sentiment=float(stock.get("news_sentiment", 0.0)),
            stop_price=float(risk.stop_price),
            take_price=float(risk.take_price),
            atr_at_entry=float(risk.atr) if risk.atr is not None else None,
            stop_basis=risk.basis,
            exec_arm=arm,
        )
        (longs if side == "long" else shorts).append(pos)
        placed_notional += order_notional
        if side == "short":
            placed_short += order_notional

        # Persist THIS leg before placing the next order. An exception on
        # stock N must never leave stocks 1..N-1 live at the broker but
        # absent from the registry (they would have no stops and would be
        # invisible to the close leg).
        if not dry_run:
            if not tranche_created:
                registry.add_tranche(
                    tranche_id=tranche_id,
                    open_date=_today_str(),
                    hold_days=config_trading.HOLD_DAYS,
                    signal_file=Path(signals.get("data_date", "")).name
                               if signals.get("data_date") else None,
                )
                tranche_created = True
            registry.add_position(tranche_id, pos)
            registry.save()

    if not longs and not shorts:
        print(f"\n[open] All positions skipped -- no tranche created.")
        return None

    print(f"\n[open] Tranche {tranche_id}: "
          f"{len(longs)} longs + {len(shorts)} shorts, "
          f"capital deployed=${sum(p.entry_amount_usd for p in longs+shorts):,.2f}")
    return tranche_id


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Staggered multi-day trading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print orders + registry actions but don't actually trade.")
    parser.add_argument("--close-only", action="store_true",
                        help="Only close due tranches, don't open new.")
    parser.add_argument("--open-only", action="store_true",
                        help="Only open new tranche, don't close due.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"STAGGERED TRADING -- {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"Config: {config_trading.summary()}")
    if args.dry_run:
        print("** DRY RUN -- no orders will be placed **")

    if config_trading.STRATEGY != "staggered":
        print(f"\n[abort] config_trading.STRATEGY = "
              f"{config_trading.STRATEGY!r} (not 'staggered'). "
              f"Use run_auto_trading.py for legacy 1-day strategy.")
        return

    trader = AlpacaAutoTrader()
    registry = TrancheRegistry(ROOT / config_trading.TRANCHE_REGISTRY_PATH)

    summary_pre = registry.summary()
    print(f"\nRegistry pre: {summary_pre}")

    # P2: converge registry to broker truth before acting (bracket legs may
    # have fired overnight; the monitor may have force-closed something).
    print("\n[reconcile] checking registry vs broker positions...")
    reconcile_registry_with_broker(trader, registry, dry_run=args.dry_run)

    n_closed = 0
    if not args.open_only:
        n_closed = close_due_tranches(trader, registry, dry_run=args.dry_run)
        time.sleep(2)

    new_tranche_id = None
    if not args.close_only:
        signals = load_today_signals()
        if signals:
            new_tranche_id = open_new_tranche(trader, registry,
                                              signals, dry_run=args.dry_run)

    if not args.dry_run:
        registry.save()

    summary_post = registry.summary()
    print(f"\n{'=' * 70}\nSUMMARY")
    print("=" * 70)
    print(f"  closed: {n_closed} order(s)")
    print(f"  opened: {'tranche ' + new_tranche_id if new_tranche_id else 'none'}")
    print(f"  registry: pre={summary_pre} -> post={summary_post}")
    print()


if __name__ == "__main__":
    main()
