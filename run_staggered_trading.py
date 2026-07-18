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

        realized_total = 0.0
        for pos in tranche.longs:
            # long -> sell
            ok = _close_position(trader, pos.symbol, pos.qty, "sell",
                                 client_order_id=f"close_{tranche.id}_{pos.symbol}",
                                 dry_run=dry_run)
            n_orders += int(ok)
            if ok:
                # P&L estimate based on current price; settlement may differ slightly
                cur = trader.get_current_price(pos.symbol)
                if cur:
                    pnl = (cur - pos.entry_price) * pos.qty
                    realized_total += pnl
                    print(f"  [SELL  long]  {pos.symbol:<6} qty={pos.qty:>4}  "
                          f"entry=${pos.entry_price:>7.2f}  cur=${cur:>7.2f}  "
                          f"P&L=${pnl:+.2f}")
        for pos in tranche.shorts:
            # short -> buy to cover
            ok = _close_position(trader, pos.symbol, pos.qty, "buy",
                                 client_order_id=f"close_{tranche.id}_{pos.symbol}",
                                 dry_run=dry_run)
            n_orders += int(ok)
            if ok:
                cur = trader.get_current_price(pos.symbol)
                if cur:
                    pnl = (pos.entry_price - cur) * pos.qty
                    realized_total += pnl
                    print(f"  [BUY  short]  {pos.symbol:<6} qty={pos.qty:>4}  "
                          f"entry=${pos.entry_price:>7.2f}  cur=${cur:>7.2f}  "
                          f"P&L=${pnl:+.2f}")

        if not dry_run:
            registry.mark_closed(
                tranche.id, realized_total, reason="scheduled_close"
            )

    return n_orders


def _close_position(trader: AlpacaAutoTrader, symbol: str, qty: int, side: str,
                    client_order_id: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [DRY-RUN] would close: {side.upper()} {qty} {symbol}")
        return True
    return trader.place_market_order(symbol, qty, side)


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

    print(f"\n{'=' * 70}\nOPENING NEW TRANCHE {tranche_id}\n{'=' * 70}")
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

        client_id = f"open_{tranche_id}_{symbol}"
        order_side = "buy" if side == "long" else "sell"

        # Compute per-position stop & take prices BEFORE the order so we can
        # log them alongside the entry. Risk levels are persisted with the
        # tranche so the monitor reads exact prices, not recomputed thresholds.
        risk = compute_risk_levels(symbol, float(cur_price), side)

        if dry_run:
            ok = True
            print(f"  [DRY-RUN] {order_side.upper():<4} {side:<5} "
                  f"{symbol:<6} qty={qty:>4} @ ${cur_price:>7.2f} "
                  f"= ${qty*cur_price:>9,.2f}  "
                  f"stop=${risk.stop_price:>7.2f}({risk.stop_pct*100:.1f}%) "
                  f"take=${risk.take_price:>7.2f}({risk.take_pct*100:.1f}%) "
                  f"[{risk.basis}]")
        else:
            ok = trader.place_market_order(symbol, qty, order_side)
            if ok:
                print(f"  [OK]      {order_side.upper():<4} {side:<5} "
                      f"{symbol:<6} qty={qty:>4} @ ${cur_price:>7.2f} "
                      f"= ${qty*cur_price:>9,.2f}  "
                      f"stop=${risk.stop_price:>7.2f}({risk.stop_pct*100:.1f}%) "
                      f"take=${risk.take_price:>7.2f}({risk.take_pct*100:.1f}%) "
                      f"[{risk.basis}]")
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
        )
        (longs if side == "long" else shorts).append(pos)

    if not longs and not shorts:
        print(f"\n[open] All positions skipped -- no tranche created.")
        return None

    if not dry_run:
        registry.add_tranche(
            tranche_id=tranche_id,
            open_date=_today_str(),
            hold_days=config_trading.HOLD_DAYS,
            longs=longs,
            shorts=shorts,
            signal_file=Path(signals.get("data_date", "")).name
                       if signals.get("data_date") else None,
        )

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
