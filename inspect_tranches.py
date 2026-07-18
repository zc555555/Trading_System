"""
Inspect / debug the staggered-trading tranche registry.

CLI:
    python inspect_tranches.py             # human-readable summary
    python inspect_tranches.py --json      # raw JSON dump
    python inspect_tranches.py --due       # only tranches due to close today
    python inspect_tranches.py --open      # only currently open tranches
    python inspect_tranches.py --closed    # only closed tranches
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config_trading
from trading.tranche_registry import TrancheRegistry
from trading.trading_calendar import today, _to_date


def fmt_money(x: float | None) -> str:
    if x is None:
        return "      n/a"
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):>9,.2f}"


def render_tranche(t, today_d):
    age_days = (today_d - _to_date(t.open_date)).days
    days_to_close = (_to_date(t.scheduled_close_date) - today_d).days
    deployed = sum(p.entry_amount_usd for p in t.longs + t.shorts)

    marker = "OPEN" if t.status == "open" else t.status.upper()
    print(f"\n  [{marker:<10}] {t.id}")
    print(f"            open={t.open_date}  sched_close={t.scheduled_close_date}  "
          f"age={age_days}d  days_to_close={days_to_close}d")
    print(f"            positions: {len(t.longs)} long / {len(t.shorts)} short, "
          f"deployed=${deployed:>9,.2f}")
    if t.status == "closed":
        print(f"            closed_at={t.closed_at}  reason={t.close_reason}  "
              f"realized={fmt_money(t.realized_pnl)}")
    if t.longs or t.shorts:
        for p in t.longs:
            print(f"               long  {p.symbol:<6} qty={p.qty:>4} "
                  f"entry=${p.entry_price:>7.2f} (=${p.entry_amount_usd:>9,.2f})")
        for p in t.shorts:
            print(f"               short {p.symbol:<6} qty={p.qty:>4} "
                  f"entry=${p.entry_price:>7.2f} (=${p.entry_amount_usd:>9,.2f})")


def main():
    parser = argparse.ArgumentParser(description="Inspect tranche registry")
    parser.add_argument("--json", action="store_true",
                        help="Dump raw JSON instead of pretty output.")
    parser.add_argument("--due", action="store_true",
                        help="Show only tranches due to close on/before today.")
    parser.add_argument("--open", action="store_true",
                        help="Show only currently open tranches.")
    parser.add_argument("--closed", action="store_true",
                        help="Show only closed tranches.")
    args = parser.parse_args()

    path = Path(__file__).parent / config_trading.TRANCHE_REGISTRY_PATH
    if not path.exists():
        print(f"[empty] No registry at {path}. Run run_staggered_trading.py first.")
        return

    registry = TrancheRegistry(path)
    today_d = today()

    if args.json:
        print(json.dumps(registry._payload, indent=2))
        return

    print(f"Registry: {path}")
    print(f"Today: {today_d}")
    print(f"Strategy: {config_trading.summary()}")
    print()
    print(f"Summary: {registry.summary()}")

    tranches = registry.tranches
    if args.due:
        tranches = registry.tranches_due_to_close(today_d)
        print(f"\nDue today (<= {today_d}): {len(tranches)} tranche(s)")
    elif args.open:
        tranches = registry.open_tranches()
        print(f"\nOpen: {len(tranches)} tranche(s)")
    elif args.closed:
        tranches = [t for t in tranches if t.status != "open"]
        print(f"\nClosed: {len(tranches)} tranche(s)")
    else:
        print(f"\nAll tranches: {len(tranches)}")

    if not tranches:
        print("  (none)")
        return

    for t in tranches:
        render_tranche(t, today_d)


if __name__ == "__main__":
    main()
