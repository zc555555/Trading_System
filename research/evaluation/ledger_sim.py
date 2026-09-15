"""Cash-and-shares ledger simulator of the staggered-tranche book
(2026-09-14 review item 2).

The construction grid's simulator marked a FIXED WEIGHT per day and
compounded the marks, i.e. it rebalanced to target weights daily; the live
book holds a fixed number of SHARES for the holding period, sizes tranches
from equity, rounds to whole shares, refuses entries the no-debt guards
block, and exits through broker-side brackets. This simulator does what
the orchestrator does, with an explicit cash account:

  * signal at the close of session t; entries fill at the OPEN of t+1;
    quantity = floor(alloc / close_t), the price the orchestrator sizes on;
  * tranche capital = equity(close_t) x capital_per_tranche_pct; per-name
    cap = per_stock_max_pct of it; guards in rank order exactly as
    trading/execution.entry_allowed: no leverage (gross + notional <=
    equity), single short <= short_single_max_pct of equity incl. existing,
    gross short <= short_gross_max_pct (counting entries placed earlier in
    the same run);
  * scheduled close at the OPEN of entry session + hold_days (the
    orchestrator closes the evening the tranche is due, fills next open);
  * brackets: stop_pct = clamp(stop_atr x ATR14 / entry, floor, ceiling),
    take_pct likewise (trading/risk_levels.py), ATR14 = 14-session mean
    true range through the signal session; a stop fills at its price, or at
    the open when the session gapped through it; a take-profit fills at its
    price, or at the open when the open was already beyond it; when both
    levels are hit in one session the stop is assumed first;
  * costs: cost_bp ROUND TRIP, half per side, on traded notional;
  * a name with no price on its exit session exits the next session it
    trades; a name that never trades again exits at its last close less
    `delist_haircut` (delisting);
  * equity = cash + long market value - short liability at every close;
    no borrow fees, no interest on cash (documented simplifications).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LedgerParams:
    hold_days: int = 20
    capital_per_tranche_pct: float = 5.0          # 100 / hold_days in config_trading
    per_stock_max_pct: float = 30.0               # of tranche capital
    short_single_max_pct: float = 2.0             # of equity, incl. existing
    short_gross_max_pct: float = 25.0             # of equity
    allow_shorts: bool = True
    apply_caps: bool = True
    cost_bp: float = 15.0                         # round trip
    initial_equity: float = 100_000.0
    integer_shares: bool = True
    brackets: bool = True
    stop_atr: float = 3.0
    take_atr: float = 6.0
    atr_window: int = 14
    stop_floor: float = 0.015
    stop_ceiling: float = 0.12
    take_floor: float = 0.025
    take_ceiling: float = 0.25
    delist_haircut: float = 0.0


@dataclass
class _Pos:
    symbol: str
    j: int
    side: int                # +1 long, -1 short
    qty: float
    entry_px: float
    entry_i: int
    exit_i: int
    stop: Optional[float]
    take: Optional[float]
    tranche: str


@dataclass
class SimResult:
    equity: pd.Series
    returns: pd.Series
    fills: pd.DataFrame
    blocked: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def atr_panel(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR as trading/atr_loader computes it: mean true range over `window`
    sessions per symbol (NaN until the window is full). Aligned to prices.index."""
    px = prices.sort_values(["symbol", "date"], kind="mergesort")
    prev_close = px.groupby("symbol", sort=False)["close"].shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - prev_close).abs(), (px["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    atr = tr.groupby(px["symbol"], sort=False).transform(lambda s: s.rolling(window, min_periods=window).mean())
    return atr.reindex(prices.index)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def simulate_ledger(books: dict, prices: pd.DataFrame, params: LedgerParams = LedgerParams()) -> SimResult:
    """books: {signal_date -> DataFrame[symbol, side ('long'|'short'),
    position_pct]} in rank order (strategy/selection.select_book output);
    prices: [date, symbol, open, high, low, close] for every session."""
    px = prices[["date", "symbol", "open", "high", "low", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    sessions = np.sort(px["date"].unique())
    sidx = {d: i for i, d in enumerate(sessions)}
    symbols = np.sort(px["symbol"].unique())
    jidx = {s: j for j, s in enumerate(symbols)}
    T, S = len(sessions), len(symbols)

    def wide(col):
        w = px.pivot_table(index="date", columns="symbol", values=col, aggfunc="first").reindex(index=sessions, columns=symbols)
        return w.to_numpy(dtype=float)

    O, H, L, C = wide("open"), wide("high"), wide("low"), wide("close")
    atr_w = None
    if params.brackets:
        px["_atr"] = atr_panel(px, params.atr_window)
        atr_w = px.pivot_table(index="date", columns="symbol", values="_atr", aggfunc="first").reindex(index=sessions, columns=symbols).to_numpy(dtype=float)
    valid = ~np.isnan(C)
    last_valid = np.full(S, -1)
    for j in range(S):
        idx = np.flatnonzero(valid[:, j])
        last_valid[j] = idx[-1] if len(idx) else -1

    cash = float(params.initial_equity)
    positions: list[_Pos] = []
    last_px = np.full(S, np.nan)
    equity = np.full(T, np.nan)
    fills: list[dict] = []
    blocked: Counter = Counter()
    half_cost = params.cost_bp / 2.0 / 1e4
    n_tranches = 0

    def mark(i: int) -> float:
        val = cash
        for p in positions:
            c = C[i, p.j]
            if not np.isnan(c):
                last_px[p.j] = c
            val += p.side * p.qty * (last_px[p.j] if not np.isnan(last_px[p.j]) else p.entry_px)
        return val

    def close_position(p: _Pos, i: int, price: float, kind: str):
        nonlocal cash
        notional = p.qty * price
        cash += p.side * notional                      # long: proceeds in; short: buy back out
        cost = notional * half_cost
        cash -= cost
        fills.append({"date": sessions[i], "symbol": p.symbol, "side": "long" if p.side > 0 else "short",
                      "qty": p.qty, "price": price, "kind": kind, "tranche": p.tranche, "cost": cost,
                      "pnl": p.side * p.qty * (price - p.entry_px)})

    equity[0] = mark(0) if T else cash
    for i in range(1, T):
        equity_prev = equity[i - 1]
        # 1. scheduled closes and delistings at the open
        keep = []
        for p in positions:
            if p.exit_i <= i:
                o = O[i, p.j]
                if not np.isnan(o):
                    close_position(p, i, o, "close")
                    continue
                if i > last_valid[p.j]:                 # never trades again: delisted
                    lp = C[last_valid[p.j], p.j] if last_valid[p.j] >= 0 else p.entry_px
                    close_position(p, i, lp * (1 - params.delist_haircut * p.side), "delist")
                    continue
            keep.append(p)
        positions = keep

        # 2. new tranche at the open, from the book decided at yesterday's close
        book = books.get(sessions[i - 1])
        if book is not None and len(book):
            tranche_capital = equity_prev * params.capital_per_tranche_pct / 100.0
            per_stock_max = params.per_stock_max_pct / 100.0 * tranche_capital
            # exposure at yesterday's close, as the orchestrator sees it in the evening
            gross = sum(abs(p.qty) * (last_px[p.j] if not np.isnan(last_px[p.j]) else p.entry_px) for p in positions)
            short_tot = sum(p.qty * (last_px[p.j] if not np.isnan(last_px[p.j]) else p.entry_px) for p in positions if p.side < 0)
            short_by: dict[str, float] = {}
            for p in positions:
                if p.side < 0:
                    short_by[p.symbol] = short_by.get(p.symbol, 0.0) + p.qty * (last_px[p.j] if not np.isnan(last_px[p.j]) else p.entry_px)
            n_tranches += 1
            tid = f"t{n_tranches}"
            for _, r in book.iterrows():
                sym = r["symbol"]
                j = jidx.get(sym)
                if j is None:
                    blocked["no_price"] += 1
                    continue
                side = 1 if r["side"] == "long" else -1
                if side < 0 and not params.allow_shorts:
                    blocked["shorts_disabled"] += 1
                    continue
                sig_px = C[i - 1, j]
                if np.isnan(sig_px) or sig_px <= 0:
                    blocked["no_price"] += 1
                    continue
                alloc = min(tranche_capital * float(r["position_pct"]) / 100.0, per_stock_max)
                qty = math.floor(alloc / sig_px) if params.integer_shares else alloc / sig_px
                if qty < (1 if params.integer_shares else 1e-9):
                    blocked["qty<1"] += 1
                    continue
                notional = qty * sig_px
                if params.apply_caps:
                    if gross + notional > equity_prev:
                        blocked["leverage"] += 1
                        continue
                    if side < 0:
                        if short_by.get(sym, 0.0) + notional > equity_prev * params.short_single_max_pct / 100.0:
                            blocked["short_single"] += 1
                            continue
                        if short_tot + notional > equity_prev * params.short_gross_max_pct / 100.0:
                            blocked["short_gross"] += 1
                            continue
                o = O[i, j]
                if np.isnan(o) or o <= 0:
                    blocked["no_open"] += 1
                    continue
                # fill
                fill_notional = qty * o
                cash -= side * fill_notional
                cost = fill_notional * half_cost
                cash -= cost
                gross += notional
                if side < 0:
                    short_tot += notional
                    short_by[sym] = short_by.get(sym, 0.0) + notional
                stop = take = None
                if params.brackets and atr_w is not None:
                    a = atr_w[i - 1, j]
                    if np.isnan(a) or a <= 0:
                        stop_pct, take_pct = 0.05, 0.10          # risk_levels fixed-pct fallback
                    else:
                        stop_pct = _clamp(params.stop_atr * a / o, params.stop_floor, params.stop_ceiling)
                        take_pct = _clamp(params.take_atr * a / o, params.take_floor, params.take_ceiling)
                    stop = o * (1 - side * stop_pct)
                    take = o * (1 + side * take_pct)
                positions.append(_Pos(sym, j, side, qty, o, i, i + params.hold_days, stop, take, tid))
                last_px[j] = o
                fills.append({"date": sessions[i], "symbol": sym, "side": "long" if side > 0 else "short", "qty": qty,
                              "price": o, "kind": "entry", "tranche": tid, "cost": cost, "pnl": 0.0})

        # 3. brackets during the session
        if params.brackets:
            keep = []
            for p in positions:
                h, l, o = H[i, p.j], L[i, p.j], O[i, p.j]
                if np.isnan(h) or np.isnan(l):
                    keep.append(p)
                    continue
                exit_px, kind = None, None
                if p.side > 0:
                    if l <= p.stop:
                        exit_px, kind = (o if (not np.isnan(o) and o < p.stop) else p.stop), "stop"
                    elif h >= p.take:
                        exit_px, kind = (o if (not np.isnan(o) and o > p.take) else p.take), "take"
                else:
                    if h >= p.stop:
                        exit_px, kind = (o if (not np.isnan(o) and o > p.stop) else p.stop), "stop"
                    elif l <= p.take:
                        exit_px, kind = (o if (not np.isnan(o) and o < p.take) else p.take), "take"
                if exit_px is None:
                    keep.append(p)
                else:
                    close_position(p, i, float(exit_px), kind)
            positions = keep

        # 4. mark
        equity[i] = mark(i)

    eq = pd.Series(equity, index=pd.DatetimeIndex(sessions), name="equity")
    rets = eq.pct_change().dropna()
    fills_df = pd.DataFrame(fills)
    stats = {"n_tranches": n_tranches, "n_fills": len(fills_df),
             "n_entries": int((fills_df["kind"] == "entry").sum()) if len(fills_df) else 0,
             "exits": fills_df["kind"].value_counts().to_dict() if len(fills_df) else {},
             "traded_notional": float((fills_df["qty"] * fills_df["price"]).sum()) if len(fills_df) else 0.0,
             "final_equity": float(eq.iloc[-1]) if len(eq) else cash}
    return SimResult(eq, rets, fills_df, dict(blocked), stats)


def books_from_predictions(panel: pd.DataFrame, pred_col: str, select_params, vol_col: str = "volatility_20d") -> dict:
    """{date -> book} by running strategy/selection.select_book on each
    date's candidates (rows with a prediction)."""
    from strategy import selection as sel
    books = {}
    cols = ["symbol", pred_col] + ([vol_col] if vol_col in panel.columns else [])
    for d, day in panel.dropna(subset=[pred_col]).groupby("date", sort=True):
        c = day[cols].rename(columns={pred_col: "prediction"})
        b = sel.select_book(c, select_params)
        if len(b):
            books[d] = b[["symbol", "side", "position_pct"]]
    return books
