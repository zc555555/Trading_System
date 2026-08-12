"""Nightly return attribution of the LIVE paper book (descriptive only).

Reads the broker's current positions, pairs them with the two-layer risk
model (research/evaluation/risk_model.py), and decomposes the latest
session's book return into market / sector / selection components.
Appends one row per session to trading_logs/attribution_history.csv --
over months this accumulates the live answer to "how much of the P&L is
beta vs stock-picking".

Approximations (documented, absorbed by the `residual` column):
  - ex-ante weight of a position over session T is taken as
    signed_qty_now x close_{T-1} / last_equity; positions entered at T's
    open are attributed for the whole session;
  - `actual` is Alpaca's equity / last_equity - 1, which includes
    intraday fills, costs, and cash drag the model cannot see.

Idempotent: re-running on the same session overwrites that session's row.
Places no orders; analytics only.  Exit code 0 on success (including the
benign "no positions" case), nonzero on real failures so the scheduled
wrapper raises the failure toast.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.append(str(Path(__file__).resolve().parent / "research"))

PROJECT = Path(__file__).resolve().parent
PANEL_PATH = PROJECT / "research" / "data" / "stocks.parquet"
HISTORY_PATH = PROJECT / "trading_logs" / "attribution_history.csv"


def broker_positions() -> tuple[list[dict], float, float]:
    from alpaca.trading.client import TradingClient
    from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    account = client.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    positions = []
    for pos in client.get_all_positions():
        qty = abs(float(pos.qty))
        sign = -1.0 if "short" in str(pos.side).lower() else 1.0
        positions.append({"symbol": pos.symbol, "signed_qty": sign * qty})
    return positions, equity, last_equity


def main() -> int:
    positions, equity, last_equity = broker_positions()
    if not positions:
        print("[attribution] no open positions; nothing to attribute")
        return 0
    if last_equity <= 0:
        print("[attribution] last_equity <= 0, cannot compute weights")
        return 1

    from evaluation.risk_model import build_risk_model, decompose_portfolio

    prices = pd.read_parquet(PANEL_PATH, columns=["date", "symbol", "close"])
    sessions = sorted(prices["date"].unique())
    if len(sessions) < 2:
        print("[attribution] price panel too short")
        return 1
    t, t_prev = sessions[-1], sessions[-2]
    prev_close = (prices[prices["date"] == t_prev]
                  .set_index("symbol")["close"].to_dict())

    rows, skipped = [], []
    for p in positions:
        pc = prev_close.get(p["symbol"])
        if pc is None or pc <= 0:
            skipped.append(p["symbol"])
            continue
        rows.append({"date": t, "symbol": p["symbol"],
                     "weight": p["signed_qty"] * pc / last_equity})
    if skipped:
        print(f"[attribution] {len(skipped)} positions missing prior close, "
              f"skipped: {skipped}")
    if not rows:
        print("[attribution] no positions matched the price panel")
        return 1
    weights = pd.DataFrame(rows)

    rm = build_risk_model()
    actual = pd.Series({t: equity / last_equity - 1.0})
    attr = decompose_portfolio(rm, weights, actual_daily=actual)
    if attr.empty:
        print(f"[attribution] session {t} not in risk model returns yet")
        return 1
    row = attr.iloc[-1]

    record = {
        "date": pd.Timestamp(t).strftime("%Y-%m-%d"),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "n_positions": len(rows),
        "actual": round(float(row["actual"]), 6),
        "market": round(float(row["market"]), 6),
        "sector": round(float(row["sector"]), 6),
        "selection": round(float(row["selection"]), 6),
        "residual": round(float(row["residual"]), 6),
        "beta_exposure": round(float(row["beta_exposure"]), 4),
        "net_weight": round(float(row["net_weight"]), 4),
        "gross_weight": round(float(row["gross_weight"]), 4),
        "equity": round(equity, 2),
    }

    if HISTORY_PATH.exists():
        hist = pd.read_csv(HISTORY_PATH)
        hist = hist[hist["date"] != record["date"]]
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])
    hist.sort_values("date").to_csv(HISTORY_PATH, index=False)

    print(f"[attribution] {record['date']}: actual {record['actual']:+.4%} = "
          f"market {record['market']:+.4%} + sector {record['sector']:+.4%} + "
          f"selection {record['selection']:+.4%} + residual {record['residual']:+.4%}"
          f"  (beta {record['beta_exposure']:+.2f}, gross {record['gross_weight']:.0%},"
          f" {record['n_positions']} positions)")
    print(f"[attribution] appended -> {HISTORY_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- unattended: fail loud, not silent
        print(f"[attribution] FAILED: {exc}")
        sys.exit(1)
