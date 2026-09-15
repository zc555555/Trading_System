"""evaluation/ledger_sim.py: a cash-and-shares book that behaves like the
orchestrator (2026-09-14 review item 2). Synthetic prices, hand-computed
expectations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "research"))

from evaluation import ledger_sim as ls  # noqa: E402

D = pd.bdate_range("2026-01-05", periods=12)


def _prices(closes: dict, opens: dict | None = None, highs=None, lows=None):
    rows = []
    for sym, cs in closes.items():
        for i, c in enumerate(cs):
            if c is None:
                continue
            o = (opens or {}).get(sym, cs)[i] if opens and sym in opens else c
            h = (highs or {}).get(sym, [None] * len(cs))[i] if highs and sym in highs else max(o, c)
            l = (lows or {}).get(sym, [None] * len(cs))[i] if lows and sym in lows else min(o, c)
            rows.append({"date": D[i], "symbol": sym, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def _book(*legs):
    return pd.DataFrame([{"symbol": s, "side": side, "position_pct": pct} for s, side, pct in legs])


NO_BRACKETS = ls.LedgerParams(hold_days=2, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0,
                              cost_bp=0.0, brackets=False)


def test_fixed_shares_round_trip_100_110_100_is_zero_not_positive():
    """The review's reproduction: fixed-weight daily compounding reported
    +0.19% on 100 -> 110 -> 100; holding shares must report exactly zero."""
    px = _prices({"AAA": [100, 100, 110, 100, 100, 100]})
    res = ls.simulate_ledger({D[0]: _book(("AAA", "long", 100.0))}, px, NO_BRACKETS)
    # entry at open of D1 (100), 1000 shares; scheduled close at open of D3 (100)
    assert res.stats["n_entries"] == 1 and res.fills.iloc[0]["qty"] == 1000
    assert res.equity.iloc[2] == pytest.approx(110_000)                # marked at 110 in between
    assert res.equity.iloc[3] == pytest.approx(100_000) and res.equity.iloc[-1] == pytest.approx(100_000)
    assert res.fills.iloc[-1]["kind"] == "close" and res.fills.iloc[-1]["price"] == 100


def test_integer_shares_tranche_capital_and_costs():
    px = _prices({"AAA": [99.5] * 12})
    p = ls.LedgerParams(hold_days=3, capital_per_tranche_pct=5.0, per_stock_max_pct=100.0, cost_bp=20.0, brackets=False)
    res = ls.simulate_ledger({D[0]: _book(("AAA", "long", 100.0))}, px, p)
    e = res.fills.iloc[0]
    assert e["qty"] == 50 and e["price"] == 99.5                        # floor(5000 / 99.5)
    assert e["cost"] == pytest.approx(50 * 99.5 * 10 / 1e4)              # half the round trip per side
    assert res.stats["final_equity"] == pytest.approx(100_000 - 2 * 50 * 99.5 * 10 / 1e4)
    # per-name cap: 30% of the tranche
    p2 = ls.LedgerParams(hold_days=3, capital_per_tranche_pct=5.0, per_stock_max_pct=30.0, cost_bp=0.0, brackets=False)
    res2 = ls.simulate_ledger({D[0]: _book(("AAA", "long", 100.0))}, px, p2)
    assert res2.fills.iloc[0]["qty"] == 15                              # floor(1500 / 99.5)


def test_short_caps_and_no_leverage_block_entries_in_rank_order():
    px = _prices({"SSS": [50.0] * 12, "TTT": [50.0] * 12, "LLL": [100.0] * 12})
    p = ls.LedgerParams(hold_days=5, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0, brackets=False,
                        short_single_max_pct=2.0, short_gross_max_pct=3.0)
    # 2% single-name cap: a 3% short is blocked; two 1.9% shorts exceed the 3% gross cap -> second blocked
    book = _book(("SSS", "short", 3.0), ("TTT", "short", 1.9), ("LLL", "short", 1.9), ("SSS", "short", 1.0))
    res = ls.simulate_ledger({D[0]: book}, px, p)
    assert res.blocked == {"short_single": 1, "short_gross": 1}
    assert res.fills[res.fills["kind"] == "entry"]["symbol"].tolist() == ["TTT", "SSS"]
    # no leverage: gross + notional <= equity
    book2 = _book(("LLL", "long", 60.0), ("SSS", "short", 1.0))
    res2 = ls.simulate_ledger({D[0]: book2, D[1]: book2}, px, p)    # second day's long would exceed equity
    assert res2.blocked.get("leverage") == 1
    # caps off: the same book goes through
    off = ls.LedgerParams(hold_days=5, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0,
                          brackets=False, apply_caps=False)
    assert ls.simulate_ledger({D[0]: book}, px, off).blocked == {}


def test_short_pnl_and_equity_accounting():
    px = _prices({"SSS": [50, 50, 40, 40, 45, 45, 45, 45, 45, 45, 45, 45]})
    p = ls.LedgerParams(hold_days=3, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0, brackets=False,
                        short_single_max_pct=100.0, short_gross_max_pct=100.0)
    res = ls.simulate_ledger({D[0]: _book(("SSS", "short", 100.0))}, px, p)
    # 2000 shares short at 50: equity 100k; marked 40 -> 120k; closed at open D4 = 45 -> 110k
    assert res.fills.iloc[0]["qty"] == 2000
    assert res.equity.iloc[2] == pytest.approx(120_000)
    assert res.equity.iloc[4] == pytest.approx(110_000) and res.stats["final_equity"] == pytest.approx(110_000)


def test_brackets_stop_and_take_with_gap_handling():
    # ATR window 2 for the test; entry at open D3 = 100 after a flat history (ATR = 2 from the ranges)
    closes = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    highs = [101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    lows = [99, 99, 99, 99, 93, 99, 99, 99, 99, 99, 99, 99]              # D4 dips to 93
    px = _prices({"AAA": closes}, highs={"AAA": highs}, lows={"AAA": lows})
    p = ls.LedgerParams(hold_days=8, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0,
                        brackets=True, atr_window=2, stop_atr=3.0, take_atr=6.0)
    res = ls.simulate_ledger({D[2]: _book(("AAA", "long", 100.0))}, px, p)
    ex = res.fills.iloc[-1]
    # ATR = 2 -> stop_pct 6%, stop 94: D4 low 93 <= 94 -> exit at 94 (no gap: open 100)
    assert ex["kind"] == "stop" and ex["price"] == pytest.approx(94.0) and ex["date"] == D[4]
    assert res.stats["final_equity"] == pytest.approx(94_000)
    # gap below the stop: fills at the open, not the stop
    opens = [100] * 12
    opens[4] = 90
    px2 = _prices({"AAA": closes}, opens={"AAA": opens}, highs={"AAA": highs}, lows={"AAA": lows})
    res2 = ls.simulate_ledger({D[2]: _book(("AAA", "long", 100.0))}, px2, p)
    assert res2.fills.iloc[-1]["kind"] == "stop" and res2.fills.iloc[-1]["price"] == pytest.approx(90.0)
    # take-profit: ATR 2 -> take 12%: D5 high 113 -> exit at 112
    highs3 = highs.copy()
    highs3[5] = 113
    lows3 = [99] * 12
    px3 = _prices({"AAA": closes}, highs={"AAA": highs3}, lows={"AAA": lows3})
    res3 = ls.simulate_ledger({D[2]: _book(("AAA", "long", 100.0))}, px3, p)
    assert res3.fills.iloc[-1]["kind"] == "take" and res3.fills.iloc[-1]["price"] == pytest.approx(112.0)


def test_missing_exit_price_waits_and_delisting_exits_at_last_close():
    # BBB stops trading after D5 (closes None); entry at open D1, scheduled exit at open D7 -> delisted
    closes = [100, 100, 100, 100, 80, 80, None, None, None, None, None, None]
    px = _prices({"BBB": closes, "AAA": [50.0] * 12})
    p = ls.LedgerParams(hold_days=6, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0, brackets=False)
    res = ls.simulate_ledger({D[0]: _book(("BBB", "long", 100.0))}, px, p)
    ex = res.fills.iloc[-1]
    assert ex["kind"] == "delist" and ex["price"] == pytest.approx(80.0) and ex["date"] == D[7]
    assert res.stats["final_equity"] == pytest.approx(80_000)
    # a one-day price gap on the exit session: exits the next session it trades
    closes2 = [100, 100, 100, 100, 100, None, 90, 90, 90, 90, 90, 90]
    px2 = _prices({"BBB": closes2, "AAA": [50.0] * 12})
    p2 = ls.LedgerParams(hold_days=4, capital_per_tranche_pct=100.0, per_stock_max_pct=100.0, cost_bp=0.0, brackets=False)
    res2 = ls.simulate_ledger({D[0]: _book(("BBB", "long", 100.0))}, px2, p2)
    ex2 = res2.fills.iloc[-1]
    assert ex2["kind"] == "close" and ex2["date"] == D[6] and ex2["price"] == 90


def test_books_from_predictions_uses_the_selection_policy():
    from strategy import selection as sel
    panel = pd.DataFrame([{"date": D[0], "symbol": f"S{i}", "pred": (i - 5) / 100, "volatility_20d": 0.02} for i in range(12)])
    books = ls.books_from_predictions(panel, "pred", sel.SelectionParams(top_n=4, min_confidence=0.0, smooth_days=1, trend_filter=False))
    b = books[D[0]]
    assert b["symbol"].tolist() == ["S11", "S0", "S10", "S1"] and b["side"].tolist() == ["long", "short", "long", "short"]
    assert b["position_pct"].sum() == pytest.approx(100.0)


def test_atr_matches_the_production_loader_formula():
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-01", periods=40)):
        c = 100 + rng.normal()
        rows.append({"date": d, "symbol": "AAA", "open": c, "high": c + abs(rng.normal()), "low": c - abs(rng.normal()), "close": c})
    px = pd.DataFrame(rows)
    got = ls.atr_panel(px, 14)
    prev = px["close"].shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - prev).abs(), (px["low"] - prev).abs()], axis=1).max(axis=1)
    want = tr.rolling(14, min_periods=14).mean()
    pd.testing.assert_series_equal(got.reset_index(drop=True), want.reset_index(drop=True), check_names=False)
