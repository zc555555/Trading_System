"""Slippage calibration from real Alpaca paper fills (Layer 1, step 1).

Pulls every filled order since the automated book went live (2026-07-21)
and measures implementation shortfall against the simulator's execution
assumption (fill at the session open following submission).

Two entry cohorts exist and are measured differently:
  - after-close submissions (the nightly 21:00 UK runs): benchmark =
    NEXT session's official open. This is the apples-to-apples test of
    the backtest's "enter at next open" assumption.
  - intraday submissions (the 2026-07-21 validation run): benchmark =
    the quote recorded at submission (registry entry_price).

Exit legs (bracket stop/take fires) are reported separately against
their trigger prices.

Honest caveat printed with results: Alpaca PAPER fills are simulated
against NBBO with no market impact -- measured slippage is a LOWER
BOUND on real-money slippage. What this calibration validates is the
mechanics and the open-vs-fill timing gap, not impact.

Output: evaluation/results/slippage_calibration.csv + printed summary.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from alpaca.trading.requests import GetOrdersRequest  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca_trader import AlpacaAutoTrader  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent.parent / "data"
LIVE_START = datetime(2026, 7, 21, tzinfo=timezone.utc)
NY = "America/New_York"


def fetch_filled_orders(trader) -> list:
    orders = trader.trading_client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.CLOSED, after=LIVE_START, limit=500,
        nested=True))
    out = []
    for o in orders:
        if o.filled_avg_price is None or o.filled_at is None:
            continue
        out.append(o)
        for leg in (o.legs or []):
            if leg.filled_avg_price is not None and leg.filled_at is not None:
                out.append(leg)
    return out


def load_opens() -> pd.DataFrame:
    px = pd.read_parquet(DATA / "stocks.parquet",
                         columns=["date", "symbol", "open"])
    px["day"] = pd.to_datetime(px["date"]).dt.tz_convert(NY).dt.date
    return px


def official_open(px: pd.DataFrame, symbol: str, day) -> float | None:
    row = px[(px["symbol"] == symbol) & (px["day"] == day)]
    if len(row):
        return float(row["open"].iloc[0])
    try:  # fill day newer than the parquet (e.g. today) -> yfinance
        import yfinance as yf
        hist = yf.Ticker(symbol).history(start=str(day), interval="1d")
        if len(hist):
            return float(hist["Open"].iloc[0])
    except Exception:
        pass
    return None


def main():
    trader = AlpacaAutoTrader()
    orders = fetch_filled_orders(trader)
    print(f"filled orders since {LIVE_START.date()}: {len(orders)}")
    px = load_opens()

    rows = []
    for o in orders:
        filled_at = pd.Timestamp(o.filled_at).tz_convert(NY)
        submitted = pd.Timestamp(o.submitted_at).tz_convert(NY)
        fill = float(o.filled_avg_price)
        side = str(o.side).split(".")[-1].lower()
        sign = 1 if side == "buy" else -1
        kind = ("exit_" + str(o.type).split(".")[-1].lower()
                if o.order_class is None or str(o.order_class) == "OrderClass.SIMPLE"
                else ("entry" if (o.legs is not None or
                                  str(o.type).split(".")[-1].lower() == "market")
                      else "exit_" + str(o.type).split(".")[-1].lower()))
        # bracket children carry type limit/stop; parents are market entries
        if str(o.type).split(".")[-1].lower() in ("stop", "limit") and kind == "entry":
            kind = "exit_" + str(o.type).split(".")[-1].lower()

        after_close = submitted.hour >= 16 or submitted.hour < 9 or \
            (submitted.hour == 9 and submitted.minute < 30)
        bench = None
        bench_name = ""
        if kind == "entry":
            if after_close:
                bench = official_open(px, o.symbol, filled_at.date())
                bench_name = "next_open"
            else:
                bench_name = "submit_quote"  # resolved below from registry
        if bench is not None:
            slip_bp = sign * (fill - bench) / bench * 1e4
        else:
            slip_bp = np.nan
        rows.append({
            "symbol": o.symbol, "side": side, "kind": kind,
            "qty": float(o.filled_qty or 0), "fill": fill,
            "submitted_ny": str(submitted), "filled_ny": str(filled_at),
            "benchmark": bench_name, "bench_px": bench, "slip_bp": slip_bp,
        })

    df = pd.DataFrame(rows)

    # intraday entries: benchmark = registry-recorded submission quote
    import json
    reg = json.load(open(ROOT / "trading_logs" / "position_tranches.json"))
    quote_map = {}
    for t in reg["tranches"]:
        for p in t["longs"] + t["shorts"]:
            quote_map[(p["symbol"], t["open_date"])] = p["entry_price"]
    mask = (df["kind"] == "entry") & (df["benchmark"] == "submit_quote")
    for i in df[mask].index:
        day = pd.Timestamp(df.at[i, "filled_ny"]).date().isoformat()
        q = quote_map.get((df.at[i, "symbol"], day))
        if q:
            sign = 1 if df.at[i, "side"] == "buy" else -1
            df.at[i, "bench_px"] = q
            df.at[i, "slip_bp"] = sign * (df.at[i, "fill"] - q) / q * 1e4

    # Execution A/B: map fills to their arm via the registry's
    # client_order_id ('open_...' = limit arm original; '..._mkt' =
    # converted-to-market after timeout; market-arm ids carry no suffix
    # but the registry's exec_arm field is authoritative).
    arm_map = {}
    for t in reg["tranches"]:
        for p in t["longs"] + t["shorts"]:
            if p.get("client_order_id"):
                arm_map[p["client_order_id"]] = p.get("exec_arm") or "pre_ab"
    df["exec_arm"] = [
        arm_map.get(getattr(o, "client_order_id", None) or "",
                    arm_map.get(((getattr(o, "client_order_id", None) or "")
                                 .removesuffix("_mkt")), None))
        for o in orders]

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "slippage_calibration.csv", index=False)

    print("\n" + "=" * 78)
    print("IMPLEMENTATION SHORTFALL vs MODEL ASSUMPTIONS  (positive = paid more)")
    print("=" * 78)

    ab = df[(df["kind"] == "entry") & df["exec_arm"].isin(["market", "limit"])]
    if len(ab):
        print("\nEXECUTION A/B (entries with an arm assignment):")
        for arm, g in ab.groupby("exec_arm"):
            with_slip = g.dropna(subset=["slip_bp"])
            print(f"  {arm:<7} n={len(g):>3}  "
                  f"mean {with_slip['slip_bp'].mean():+7.2f} bp  "
                  f"median {with_slip['slip_bp'].median():+7.2f} bp"
                  if len(with_slip) else f"  {arm:<7} n={len(g):>3}  (no benchmark)")
    for (kind, bench), g in df.dropna(subset=["slip_bp"]).groupby(["kind", "benchmark"]):
        print(f"\n{kind} vs {bench}  (n={len(g)}):")
        print(f"  mean {g['slip_bp'].mean():+7.2f} bp   "
              f"median {g['slip_bp'].median():+7.2f} bp   "
              f"worst {g['slip_bp'].max():+7.2f} bp")
        for _, r in g.iterrows():
            print(f"    {r['symbol']:<6} {r['side']:<4} "
                  f"fill={r['fill']:<9.2f} bench={r['bench_px']:<9.2f} "
                  f"{r['slip_bp']:+7.2f} bp")
    n_na = df["slip_bp"].isna().sum()
    if n_na:
        print(f"\n[note] {n_na} orders without benchmark (see CSV)")
    print("\n[caveat] paper fills simulate NBBO without impact -- these numbers"
          "\nare a LOWER BOUND on live slippage; they validate timing vs the"
          "\nbacktest's next-open assumption, not market impact.")
    print(f"saved: {RESULTS / 'slippage_calibration.csv'}")


if __name__ == "__main__":
    main()
