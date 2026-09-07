"""
Staggered multi-day trading orchestrator (W2-B, 2026-05; execution safety
rewrite 2026-09 -- phase 1 of the review).

Each invocation runs ONE daily cycle:
    0. Refuse to add risk while the account is HALTED (trading/halt.py) and
       evaluate the daily-loss breaker itself (it no longer depends on the
       monitor process).
    1. Reconcile the registry with the broker. A failed broker query ABORTS
       the cycle (exit code 2, nothing touched); a broker shortfall reduces
       quantities oldest-tranche-first; positions the registry does not know
       are reported as orphans and never traded.
    2. Close any tranches whose scheduled_close_date <= today. A leg is only
       removed for the quantity that actually FILLED; partial fills and
       still-queued closes stay registered for the next run. Retries reuse
       the client order id (suffixed) so a timeout can never duplicate an
       order.
    3. Open a NEW tranche sized by today's signals, with the no-debt guards
       applied to the cumulative book: existing positions + resting entry
       orders + this order, per symbol and in aggregate.
    4. Persist the registry after every leg.

CLI:
    python run_staggered_trading.py
    python run_staggered_trading.py --dry-run     # don't send orders
    python run_staggered_trading.py --close-only  # close due tranches, skip new
    python run_staggered_trading.py --open-only   # open new, skip closes
    python run_staggered_trading.py --clear-halt  # human: lift a halt, then exit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import config_trading
from trading import execution as ex
from trading import halt
from trading import trading_calendar as cal
from trading.broker import BrokerError, TERMINAL
from trading.tranche_registry import TrancheRegistry, TranchePosition
from trading.risk_levels import compute_risk_levels
from trading.atr_loader import atr_summary

ROOT = Path(__file__).parent
FAILURES_LOG = ROOT / "trading_logs" / "FAILURES.log"
EXEC_ARM_LOG = ROOT / "trading_logs" / "exec_arms.csv"
EXIT_BROKER_UNAVAILABLE = 2


def _now_et() -> datetime:
    """Exchange-local clock: tranche dates and 'today' follow New York, not
    the machine's zone (they agree at the scheduled 21:00 UK run, not
    necessarily when run by hand)."""
    return cal.eastern_now()


def _today_str() -> str:
    return _now_et().strftime("%Y-%m-%d")


def _today_id_stamp() -> str:
    return _now_et().strftime("%Y%m%d")


def _failure(msg: str) -> None:
    """One line in FAILURES.log (the scheduled .bat also raises a toast on a
    non-zero exit); never fatal."""
    try:
        FAILURES_LOG.parent.mkdir(exist_ok=True)
        with open(FAILURES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:                                  # noqa: BLE001
        pass


def _log_exec_arm(tranche_id: str, symbol: str, side: str, client_order_id: str,
                  arm: str, arrival_price: float) -> None:
    """Append one row per placed entry so fills can be joined to their A/B
    arm forever (the registry forgets closed legs). Never fatal."""
    try:
        EXEC_ARM_LOG.parent.mkdir(exist_ok=True)
        new = not EXEC_ARM_LOG.exists()
        with open(EXEC_ARM_LOG, "a", encoding="utf-8", newline="") as f:
            if new:
                f.write("logged_at,tranche_id,symbol,side,client_order_id,exec_arm,arrival_price\n")
            f.write(f"{datetime.now().isoformat(timespec='seconds')},{tranche_id},{symbol},"
                    f"{side},{client_order_id},{arm},{arrival_price:.4f}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] exec-arm log failed for {symbol}: {exc}")


def exec_arm_for(symbol: str, date_str: str) -> str:
    """Deterministic A/B assignment: reproducible, balanced-in-expectation,
    uncorrelated with the signal (hash of symbol+date)."""
    import hashlib
    h = hashlib.sha256(f"{symbol}|{date_str}".encode()).digest()
    return "limit" if h[0] % 2 else "market"


# ---------------------------------------------------------------------------
# RECONCILE
# ---------------------------------------------------------------------------
def reconcile_registry_with_broker(broker, registry: TrancheRegistry, dry_run: bool) -> ex.ReconcileReport:
    """See trading/execution.reconcile. Prints the report; the caller decides
    whether an abort stops the cycle (it must)."""
    rep = ex.reconcile(broker, registry, dry_run=dry_run)
    if rep.aborted:
        print(f"  [reconcile] ABORT: {rep.reason} -- registry untouched, no orders this cycle")
        return rep
    for tid, sym, side, q in rep.removed:
        print(f"  [reconcile] {sym} {side}: broker holds none of {tid}'s {q} -- leg removed")
    for tid, sym, side, was, now in rep.reduced:
        print(f"  [reconcile] {sym} {side}: broker holds less -- {tid} leg {was} -> {now}")
    for sym, side, bq, rq in rep.orphans:
        print(f"  [reconcile] ORPHAN {sym} {side}: broker {bq:.0f} vs registry {rq} -- not traded, check manually")
        _failure(f"ORPHAN position {sym} {side}: broker {bq:.0f} vs registry {rq}")
    for tid, sym, side, rem, otype, status in rep.pending:
        print(f"  [reconcile] {sym} {side}: {tid} entry still in flight ({otype} {status}, {rem:.0f} to fill) -- leg kept")
    if not rep.changed and not rep.orphans and not rep.pending:
        print("  [reconcile] registry matches broker")
    return rep


def expire_stale_entries(broker, registry: TrancheRegistry, now_et: datetime, dry_run: bool) -> ex.ExpireReport:
    """Limit entries resting since before the last completed session's open
    are cancelled and their registry legs dropped (see trading/execution)."""
    stale_before = datetime.combine(cal.last_completed_session(now_et), dtime(9, 30), tzinfo=cal.EASTERN)
    rep = ex.expire_stale_entries(broker, registry, stale_before, dry_run=dry_run)
    if rep.aborted:
        print(f"  [expire] ABORT: {rep.reason}")
        return rep
    for tid, sym, side, qty, filled in rep.expired:
        what = "would cancel" if dry_run else "cancelled"
        print(f"  [expire] {sym} {side}: {tid} limit entry unfilled since before {stale_before:%Y-%m-%d %H:%M} ET "
              f"-- {what}, leg {qty} -> {filled}")
        if not dry_run:
            _failure(f"ENTRY EXPIRED {sym} {side} {tid}: limit entry never filled ({filled}/{qty}); leg dropped")
    if rep.reason and not rep.aborted:
        print(f"  [expire] warning: {rep.reason}")
    return rep


# ---------------------------------------------------------------------------
# CLOSE LEG
# ---------------------------------------------------------------------------
def close_due_tranches(broker, registry: TrancheRegistry, dry_run: bool,
                       sleep=time.sleep, timeout_s: float = 45.0) -> int:
    """Close all tranches whose scheduled_close_date <= today. Only FILLED
    quantity leaves the registry."""
    due = registry.tranches_due_to_close(_today_str())
    if not due:
        print("[close] No tranches due today.")
        return 0

    print(f"\n{'=' * 70}\nCLOSING {len(due)} DUE TRANCHE(S)\n{'=' * 70}")
    n_filled_legs = 0
    holders: dict[str, int] = {}                     # symbols held by more than one open tranche
    for t in registry.open_tranches():
        for p in t.longs + t.shorts:
            holders[p.symbol] = holders.get(p.symbol, 0) + 1
    for tranche in due:
        n_pos = len(tranche.longs) + len(tranche.shorts)
        print(f"\n[close] {tranche.id}  open={tranche.open_date}  "
              f"sched_close={tranche.scheduled_close_date}  positions={n_pos}")
        legs = ([(pos, "sell", "long") for pos in tranche.longs]
                + [(pos, "buy", "short") for pos in tranche.shorts])
        realized_total = 0.0
        n_open = 0
        for pos, order_side, side in legs:
            cid = f"close_{tranche.id}_{pos.symbol}"
            if dry_run:
                print(f"  [DRY-RUN] would close: {order_side.upper()} {pos.qty} {pos.symbol}")
                continue
            res = ex.close_leg(broker, pos.symbol, int(pos.qty), order_side, client_order_id=cid,
                               timeout_s=timeout_s, sleep=sleep, entry_client_id=pos.client_order_id,
                               symbol_shared=holders.get(pos.symbol, 0) > 1)
            if res.filled <= 0:
                n_open += 1
                print(f"  [OPEN] {order_side.upper()} {pos.qty} {pos.symbol}: nothing filled "
                      f"({res.status}{': ' + res.note if res.note else ''}) -- leg kept for retry")
                continue
            # realized P&L on the filled part, from the fill price when the broker gave one
            fill_px = None
            try:
                st = broker.get_order(res.order_id) if res.order_id else None
                fill_px = st.filled_avg_price if st else None
            except BrokerError:
                pass
            px = fill_px or broker.latest_price(pos.symbol) or pos.entry_price
            pnl = ((px - pos.entry_price) if side == "long" else (pos.entry_price - px)) * res.filled
            realized_total += pnl
            tag = "[SELL  long]" if side == "long" else "[BUY  short]"
            print(f"  {tag}  {pos.symbol:<6} filled {res.filled}/{pos.qty}  entry=${pos.entry_price:>7.2f}  "
                  f"fill=${px:>7.2f}  P&L=${pnl:+.2f}{'' if res.complete else '  (PARTIAL: remainder kept)'}")
            if res.complete:
                registry.remove_position(tranche.id, pos.symbol, side, reason="scheduled_close")
                n_filled_legs += 1
            else:
                registry.reduce_position(tranche.id, pos.symbol, side, res.filled, reason="partial_close")
                n_open += 1
            registry.save()
        if dry_run:
            continue
        if n_open == 0:
            registry.mark_closed(tranche.id, realized_total, reason="scheduled_close")
            registry.save()
        else:
            print(f"  [WARN] {tranche.id}: {n_open} leg(s) not fully closed; tranche stays open, retried next run")
    return n_filled_legs


# ---------------------------------------------------------------------------
# OPEN LEG
# ---------------------------------------------------------------------------
def load_today_signals() -> dict | None:
    """Load today's signal file from research/artifacts/ (falls back to
    yesterday's so a missed overnight signal-gen doesn't kill the day)."""
    artifacts = ROOT / "research" / "artifacts"
    for offset in (0, -1):
        target = _now_et().date() + timedelta(days=offset)
        path = artifacts / f"signals_multi_factor_{target.strftime('%Y%m%d')}.json"
        if path.exists():
            print(f"[signals] Using {path.name}")
            with open(path, "r", encoding="utf-8") as f:
                signals = json.load(f)
                signals["_file"] = path.name
                return signals
    print("[signals] No signal file found for today or yesterday -- skipping new tranche.")
    return None


# ---------------------------------------------------------------------------
# DATA FRESHNESS + CALENDAR GATES (2026-09 review)
# ---------------------------------------------------------------------------
FRESH_MAX_AGE_H = 36
_DATE_RE = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2})")


def _signal_data_date(signals: dict):
    m = _DATE_RE.match(str(signals.get("data_date") or ""))
    return cal._to_date(m.group(1)) if m else None


def _tranche_data_dates(registry: TrancheRegistry) -> set:
    """data_date of every tranche ever opened (notes 'data_date=YYYY-MM-DD';
    pre-2026-09 tranches stored the raw data_date string as signal_file)."""
    out = set()
    for t in registry.tranches:
        for text in (t.notes or "", t.signal_file or ""):
            m = _DATE_RE.search(text.replace("data_date=", ""))
            if m:
                out.add(m.group(1))
    return out


def signals_fresh(signals: dict, registry: TrancheRegistry, now_et: datetime) -> tuple[bool, str]:
    """A tranche may only be opened on signals computed from the LAST
    COMPLETED session, generated recently, and not already traded. Catches
    the holiday case (Friday's data re-run on Monday), a dead data feed
    (data_date stuck), and a stale file picked up by the yesterday fallback."""
    dd = _signal_data_date(signals)
    if dd is None:
        return False, "signal file carries no data_date"
    expected = cal.last_completed_session(now_et)
    if dd != expected:
        return False, f"signal data_date {dd} != last completed session {expected} (stale or premature data)"
    gen = signals.get("generated_at")
    if gen:
        try:
            age_h = (datetime.now() - datetime.fromisoformat(str(gen)).replace(tzinfo=None)).total_seconds() / 3600
        except ValueError:
            age_h = None
        if age_h is not None and (age_h > FRESH_MAX_AGE_H or age_h < -6):
            return False, f"signal file generated {age_h:.0f}h ago (> {FRESH_MAX_AGE_H}h)"
    if dd.isoformat() in _tranche_data_dates(registry):
        return False, f"a tranche was already opened on data_date {dd}"
    return True, ""


def calendar_gate(broker, now_et: datetime) -> bool:
    """True when today (exchange time) is a trading session. The broker's
    calendar decides when it can be asked (it knows unscheduled closures);
    the built-in NYSE calendar is the fallback, and any disagreement between
    the two over the next two weeks is logged for a human."""
    d = now_et.date()
    theirs = None
    try:
        theirs = set(broker.calendar(d, d + timedelta(days=14)))
        ours = {d + timedelta(days=i) for i in range(15) if cal.is_trading_day(d + timedelta(days=i))}
        if theirs != ours:
            msg = (f"CALENDAR MISMATCH built-in vs broker: only built-in={sorted(ours - theirs)} "
                   f"only broker={sorted(theirs - ours)}")
            print(f"  [calendar] {msg}")
            _failure(msg)
    except BrokerError as e:
        print(f"  [calendar] broker calendar unavailable ({e}); using the built-in NYSE calendar")
    open_today = (d in theirs) if theirs is not None else cal.is_trading_day(d)
    print(f"[calendar] {d} {now_et.strftime('%H:%M')} ET: "
          f"{'trading session' if open_today else 'MARKET CLOSED (weekend/holiday)'}")
    return open_today


def _submit_entry(broker, symbol: str, qty: int, order_side: str, arm: str, cur_price: float,
                  stop: float, take: float, client_id: str, sleep=time.sleep) -> bool:
    """Bracket entry (market or limit arm). True when the broker has the
    order and did not reject it within a short window; an accepted entry is
    NOT a fill, which is why the entry's notional is booked as pending
    exposure until it fills."""
    try:
        st = broker.submit(symbol, qty, order_side, kind="limit" if arm == "limit" else "market",
                           limit_price=float(cur_price) if arm == "limit" else None,
                           bracket=(stop, take), client_order_id=client_id, tif="gtc")
    except BrokerError as e:
        print(f"  [FAIL]    {order_side.upper():<4} {symbol}: {e}")
        return False
    for _ in range(5):
        if st.status in TERMINAL and st.status != "filled":
            print(f"  [FAIL]    {order_side.upper():<4} {symbol}: {st.status}")
            return False
        if st.status == "filled" or st.status in ("accepted", "new", "partially_filled", "held", "pending_new"):
            return True
        sleep(1)
        try:
            st = broker.get_order(st.id)
        except BrokerError:
            return True                                   # submitted; status unknown -> treated as live
    return True


def open_new_tranche(broker, registry: TrancheRegistry, signals: dict, dry_run: bool,
                     price_of=None, sleep=time.sleep, now_et: datetime | None = None) -> str | None:
    """Build today's tranche from signals; place orders; register."""
    if not signals.get("should_trade"):
        print(f"[open] should_trade=False (n_stocks={signals.get('n_stocks', 0)}) -- no tranche opened today.")
        return None
    now_et = now_et or _now_et()
    fresh, why = signals_fresh(signals, registry, now_et)
    if not fresh:
        print(f"[open] data freshness gate: {why} -- no tranche opened.")
        _failure(f"SIGNALS NOT FRESH: {why}")
        return None
    if halt.is_halted():
        st = halt.read()
        print(f"[open] ACCOUNT HALTED since {st.get('triggered_at')} ({st.get('reason')}) -- no new risk. "
              f"Clear with --clear-halt after review.")
        return None
    try:
        account = broker.account()
    except BrokerError as e:
        print(f"[open] Cannot fetch account ({e}); aborting open-leg.")
        return None
    equity = float(account["equity"])
    last_equity = float(account.get("last_equity") or 0.0)
    if halt.daily_loss_breached(equity, last_equity, config_trading.MAX_DAILY_LOSS_PCT):
        loss = (equity - last_equity) / last_equity * 100
        if dry_run:
            print(f"[open] daily loss {loss:.2f}% breached the breaker -- would set HALT (dry run: state untouched).")
            return None
        halt.set_halt(f"daily loss {loss:.2f}% breached {config_trading.MAX_DAILY_LOSS_PCT*100:.1f}% (orchestrator)",
                      equity=equity, extra={"last_equity": last_equity})
        _failure(f"HALT set: daily loss {loss:.2f}% -- no new tranche; clear with --clear-halt")
        print(f"[open] daily loss {loss:.2f}% breached the breaker -- HALT set, no new tranche.")
        return None

    tranche_pct = config_trading.derived_capital_per_tranche_pct()
    tranche_capital = equity * (tranche_pct / 100.0)
    per_stock_max = (config_trading.PER_STOCK_MAX_PCT / 100.0) * tranche_capital
    tranche_id = f"tranche_{_today_id_stamp()}"
    if registry.get(tranche_id) is not None:
        print(f"[open] {tranche_id} already exists -- already ran today. Skipping.")
        return None

    price_of = price_of or broker.latest_price
    try:
        snap = ex.exposure(broker, price_of=price_of)   # positions + resting entries
    except BrokerError as e:
        print(f"[open] Cannot read exposure ({e}); aborting open-leg (no-debt guard needs the book).")
        return None

    print(f"\n{'=' * 70}\nOPENING NEW TRANCHE {tranche_id}\n{'=' * 70}")
    print(f"  [no-debt] gross ${snap.gross:,.0f} (short ${snap.short:,.0f}, "
          f"{snap.pending_orders} resting entries) vs equity ${equity:,.0f}")
    print(f"  tranche capital:    ${tranche_capital:>12,.2f}  ({tranche_pct:.1f}% of equity)")
    print(f"  per-stock max:      ${per_stock_max:>12,.2f}  ({config_trading.PER_STOCK_MAX_PCT}% of tranche)")
    print(f"  hold_days:          {config_trading.HOLD_DAYS}   shorts allowed: {config_trading.ALLOW_SHORTS}")
    if config_trading.USE_ATR_STOPS:
        print(f"  risk model:         ATR-adaptive (stop={config_trading.STOP_ATR_MULTIPLE}x,"
              f" take={config_trading.TAKE_ATR_MULTIPLE}x ATR_14)  [atr] {atr_summary()}")

    longs: list[TranchePosition] = []
    shorts: list[TranchePosition] = []
    tranche_created = False
    for stock in signals["stocks"]:
        symbol = stock["symbol"]
        pred = float(stock["prediction"])
        pos_pct = float(stock["position_pct"]) / 100.0
        side = "long" if pred > 0 else "short"
        if side == "short" and (config_trading.LONG_ONLY_FILTER or not config_trading.ALLOW_SHORTS):
            print(f"  [SKIP] {symbol}: SELL signal but shorts disabled")
            continue
        alloc = min(tranche_capital * pos_pct, per_stock_max)
        cur_price = price_of(symbol)
        if cur_price is None or cur_price <= 0:
            print(f"  [SKIP] {symbol}: no price")
            continue
        qty = int(alloc // cur_price)
        if qty < 1:
            print(f"  [SKIP] {symbol}: qty<1 (alloc=${alloc:.2f} / price=${cur_price:.2f})")
            continue
        order_notional = qty * cur_price
        ok, why = ex.entry_allowed(snap, symbol, side, order_notional, equity,
                                   config_trading.SHORT_SINGLE_MAX_PCT, config_trading.SHORT_GROSS_MAX_PCT)
        if not ok:
            print(f"  [BLOCK] {symbol}: {why}")
            continue
        client_id = f"open_{tranche_id}_{symbol}"
        order_side = "buy" if side == "long" else "sell"
        risk = compute_risk_levels(symbol, float(cur_price), side)
        arm = exec_arm_for(symbol, _today_str()) if config_trading.EXECUTION_AB_TEST else "market"
        line = (f"{order_side.upper():<4} {side:<5} {symbol:<6} qty={qty:>4} @ ${cur_price:>7.2f} = ${order_notional:>9,.2f}  "
                f"stop=${risk.stop_price:>7.2f}({risk.stop_pct*100:.1f}%) take=${risk.take_price:>7.2f}({risk.take_pct*100:.1f}%) "
                f"[{risk.basis}, {arm}]")
        if dry_run:
            print(f"  [DRY-RUN] {line}")
        else:
            if not _submit_entry(broker, symbol, qty, order_side, arm, cur_price, risk.stop_price, risk.take_price,
                                 client_id, sleep=sleep):
                continue
            print(f"  [OK]      {line}@broker")
        pos = TranchePosition(symbol=symbol, side=side, qty=qty, entry_price=float(cur_price),
                              entry_amount_usd=float(order_notional), client_order_id=client_id, prediction=pred,
                              sentiment=float(stock.get("news_sentiment", 0.0)),
                              stop_price=float(risk.stop_price), take_price=float(risk.take_price),
                              atr_at_entry=float(risk.atr) if risk.atr is not None else None,
                              stop_basis=risk.basis, exec_arm=arm)
        (longs if side == "long" else shorts).append(pos)
        ex.apply_entry(snap, symbol, side, order_notional)
        if not dry_run:
            _log_exec_arm(tranche_id, symbol, side, client_id, arm, float(cur_price))
            if not tranche_created:
                dd = _signal_data_date(signals)
                registry.add_tranche(tranche_id=tranche_id, open_date=_today_str(), hold_days=config_trading.HOLD_DAYS,
                                     signal_file=signals.get("_file"),
                                     notes=f"data_date={dd.isoformat()}" if dd else None)
                tranche_created = True
            registry.add_position(tranche_id, pos)
            registry.save()

    if not longs and not shorts:
        print("\n[open] All positions skipped -- no tranche created.")
        return None
    print(f"\n[open] Tranche {tranche_id}: {len(longs)} longs + {len(shorts)} shorts, "
          f"capital deployed=${sum(p.entry_amount_usd for p in longs + shorts):,.2f}")
    return tranche_id


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Staggered multi-day trading")
    parser.add_argument("--dry-run", action="store_true", help="Print orders + registry actions but don't actually trade.")
    parser.add_argument("--close-only", action="store_true", help="Only close due tranches, don't open new.")
    parser.add_argument("--open-only", action="store_true", help="Only open new tranche, don't close due.")
    parser.add_argument("--clear-halt", action="store_true", help="Lift the account halt after a human review, then exit.")
    args = parser.parse_args(argv)

    if args.clear_halt:
        st = halt.clear(by="run_staggered_trading --clear-halt")
        print(f"[halt] cleared: {st}")
        return 0

    print("=" * 70)
    print(f"STAGGERED TRADING -- {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"Config: {config_trading.summary()}")
    if args.dry_run:
        print("** DRY RUN -- no orders will be placed **")
    if config_trading.STRATEGY != "staggered":
        print(f"\n[abort] config_trading.STRATEGY = {config_trading.STRATEGY!r} (not 'staggered').")
        return 0
    if halt.is_halted():
        st = halt.read()
        print(f"\n[halt] ACCOUNT HALTED since {st.get('triggered_at')}: {st.get('reason')} -- "
              f"closes proceed, no new risk; clear with --clear-halt")

    from trading.broker import AlpacaBroker
    from alpaca_trader import AlpacaAutoTrader
    broker = AlpacaBroker()
    trader = AlpacaAutoTrader()                        # price lookup with the yfinance fallback
    registry = TrancheRegistry(ROOT / config_trading.TRANCHE_REGISTRY_PATH)
    summary_pre = registry.summary()
    print(f"\nRegistry pre: {summary_pre}")

    now_et = _now_et()
    session_today = calendar_gate(broker, now_et)

    print("\n[expire] checking for limit entries that never filled...")
    exp = expire_stale_entries(broker, registry, now_et, dry_run=args.dry_run)
    if exp.aborted:
        _failure(f"BROKER UNAVAILABLE at expire: {exp.reason} -- cycle aborted")
        return EXIT_BROKER_UNAVAILABLE

    print("\n[reconcile] checking registry vs broker positions...")
    rep = reconcile_registry_with_broker(broker, registry, dry_run=args.dry_run)
    if rep.aborted:
        _failure(f"BROKER UNAVAILABLE at reconcile: {rep.reason} -- cycle aborted")
        return EXIT_BROKER_UNAVAILABLE
    n_closed = 0
    if not args.open_only:
        if session_today:
            n_closed = close_due_tranches(broker, registry, dry_run=args.dry_run)
            time.sleep(2)
        else:
            print("[close] market closed today -- due tranches are closed at the next session")
    new_tranche_id = None
    if not args.close_only:
        if not session_today:
            print("[open] market closed today -- no signal can be fresh; nothing opened")
        else:
            signals = load_today_signals()
            if signals:
                new_tranche_id = open_new_tranche(broker, registry, signals, dry_run=args.dry_run,
                                                  price_of=trader.get_current_price, now_et=now_et)
    if not args.dry_run:
        registry.save()
    summary_post = registry.summary()
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"  closed legs (filled): {n_closed}")
    print(f"  opened: {'tranche ' + new_tranche_id if new_tranche_id else 'none'}")
    print(f"  registry: pre={summary_pre} -> post={summary_post}")
    print(f"  halt: {'YES' if halt.is_halted() else 'no'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
