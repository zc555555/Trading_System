"""Cost-aware portfolio simulation from the walk-forward prediction panel.

Execution model (matches how production actually trades, unlike the old
backtest which filled at the same close the signal was computed from):

  - Signals are computed after close of day T (features use data through T).
  - Orders are submitted that evening and FILL AT THE NEXT SESSION'S OPEN.
  - legacy mode   : exit the same session at its close (EOD flatten).
  - staggered mode: hold ``hold_days`` sessions, exit at the closing session's
                    close; 5 overlapping tranches of 1/5 equity each
                    (fixed tranche capital, no intra-tranche compounding).

Costs: commission + slippage charged per side on traded notional.
Selection mirrors production: rank by |pred|, top N=10, min 3 names else
cash, weights proportional to |pred| capped per name, direction = sign(pred).
News-sentiment adjustment is NOT simulated (no historical news archive) --
the baseline measures the model signal alone.

Usage:
    python evaluation/simulate_portfolio.py               # both modes
    python evaluation/simulate_portfolio.py --mode legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import portfolio_metrics  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stocks_with_time_windows.parquet"
HOLDOUT_START = "2025-07-01"

COMMISSION = 0.0005   # 5 bps per side
SLIPPAGE = 0.0010     # 10 bps per side
ROUND_TRIP_COST = 2 * (COMMISSION + SLIPPAGE)

TOP_N = 10
MIN_STOCKS = 3
MAX_WEIGHT_LEGACY = 0.15      # per-name cap, share of equity (alpaca_trader)
MAX_WEIGHT_TRANCHE = 0.30     # per-name cap, share of tranche (config_trading)


def _select(day: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Production selection: top-N by |pred|, |pred|-proportional capped weights."""
    day = day.assign(conf=day['pred'].abs())
    day = day.nlargest(TOP_N, 'conf')
    if len(day) < MIN_STOCKS:
        return day.iloc[0:0]
    day = day.assign(weight=(day['conf'] / day['conf'].sum()).clip(upper=max_weight),
                     direction=np.sign(day['pred']))
    return day[day['direction'] != 0]


def _load_panel_with_execution_prices(horizon: int = 1) -> pd.DataFrame:
    suffixed = RESULTS_DIR / f"oos_predictions_h{horizon}.parquet"
    legacy_name = RESULTS_DIR / "oos_predictions.parquet"
    panel = pd.read_parquet(suffixed if suffixed.exists() else legacy_name)
    prices = pd.read_parquet(DATA_PATH, columns=['date', 'symbol', 'open', 'close'])
    prices = prices.sort_values(['symbol', 'date'])
    g = prices.groupby('symbol')
    prices['next_open'] = g['open'].shift(-1)
    prices['next_close'] = g['close'].shift(-1)
    panel = panel.merge(prices[['date', 'symbol', 'next_open', 'next_close']],
                        on=['date', 'symbol'], how='left')
    return panel


def simulate_legacy(panel: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """1-day hold: enter next open, exit next close, flat overnight."""
    daily, trades = {}, []
    for date, day in panel.groupby('date', sort=True):
        day = day.dropna(subset=['pred', 'next_open', 'next_close'])
        day = day[day['next_open'] > 0]
        sel = _select(day, MAX_WEIGHT_LEGACY)
        if sel.empty:
            continue
        raw = sel['direction'] * (sel['next_close'] / sel['next_open'] - 1)
        net = raw - ROUND_TRIP_COST
        daily[date] = float((sel['weight'] * net).sum())
        for _, r in sel.iterrows():
            trades.append({'signal_date': date, 'symbol': r['symbol'],
                           'direction': 'BUY' if r['direction'] > 0 else 'SELL',
                           'weight': r['weight'],
                           'net_return': float(r['direction'] * (r['next_close'] / r['next_open'] - 1)
                                               - ROUND_TRIP_COST)})
    return pd.Series(daily).sort_index(), pd.DataFrame(trades)


def simulate_staggered(panel: pd.DataFrame, prices_wide: dict,
                       hold_days: int = 5,
                       collect: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """k-day hold, k overlapping tranches of 1/k equity.

    Daily mark for entry session = open->close; later sessions close->close;
    exit at the k-th session's close. Half the round-trip cost is charged on
    the entry mark, half on the exit mark. Tranche capital is fixed at 1/k
    (marks are additive, not compounded, within a tranche's life).

    ``collect``: optional dict; when provided it is filled with
      'weights': {date: {symbol: net_weight}} -- the book's daily net
                 weights (share of equity, shorts negative), and
      'costs':   {date: cost_drag} -- trading-cost charge booked that day.
    Used by the attribution layer; simulation output is unchanged.
    """
    sessions = prices_wide['sessions']
    sess_idx = {d: i for i, d in enumerate(sessions)}
    open_px, close_px = prices_wide['open'], prices_wide['close']

    daily = defaultdict(float)
    trades = []
    coll_w = defaultdict(lambda: defaultdict(float)) if collect is not None else None
    coll_c = defaultdict(float) if collect is not None else None
    for date, day in panel.groupby('date', sort=True):
        i = sess_idx.get(date)
        if i is None or i + 1 >= len(sessions):
            continue
        day = day.dropna(subset=['pred'])
        sel = _select(day, MAX_WEIGHT_TRANCHE)
        if sel.empty:
            continue
        last_h = min(hold_days, len(sessions) - 1 - i)
        for _, r in sel.iterrows():
            sym, w, d = r['symbol'], r['weight'] / hold_days, r['direction']
            entry = open_px.get((sessions[i + 1], sym))
            if entry is None or not np.isfinite(entry) or entry <= 0:
                continue
            prev = entry
            for h in range(1, last_h + 1):
                px = close_px.get((sessions[i + h], sym))
                if px is None or not np.isfinite(px):
                    break
                mark = d * (px / prev - 1)
                if h == 1:
                    mark -= ROUND_TRIP_COST / 2
                if h == last_h:
                    mark -= ROUND_TRIP_COST / 2
                daily[sessions[i + h]] += w * mark
                if coll_w is not None:
                    coll_w[sessions[i + h]][sym] += w * d
                    if h == 1:
                        coll_c[sessions[i + h]] += w * ROUND_TRIP_COST / 2
                    if h == last_h:
                        coll_c[sessions[i + h]] += w * ROUND_TRIP_COST / 2
                prev = px
            exit_px = close_px.get((sessions[i + last_h], sym))
            if exit_px is not None and np.isfinite(exit_px):
                trades.append({'signal_date': date, 'symbol': sym,
                               'direction': 'BUY' if d > 0 else 'SELL',
                               'weight': r['weight'],
                               'net_return': float(d * (exit_px / entry - 1)
                                                   - ROUND_TRIP_COST)})
    if collect is not None:
        collect['weights'] = {dt: dict(sy) for dt, sy in coll_w.items()}
        collect['costs'] = dict(coll_c)
    return pd.Series(dict(daily)).sort_index(), pd.DataFrame(trades)


def _report(name: str, daily: pd.Series, trades: pd.DataFrame) -> dict:
    out = {'mode': name}
    holdout_cut = pd.Timestamp(HOLDOUT_START)
    idx = pd.DatetimeIndex(daily.index)
    if idx.tz is not None:
        holdout_cut = holdout_cut.tz_localize(idx.tz)
    segments = {'all': daily,
                'dev': daily[idx < holdout_cut],
                'holdout': daily[idx >= holdout_cut]}
    for seg_name, series in segments.items():
        out[seg_name] = portfolio_metrics(series)
    if len(trades):
        t = trades.copy()
        out['trade_stats'] = {
            'n_trades': int(len(t)),
            'win_rate': float((t['net_return'] > 0).mean()),
            'avg_net_return': float(t['net_return'].mean()),
            'buy_win_rate': float((t.loc[t['direction'] == 'BUY', 'net_return'] > 0).mean())
            if (t['direction'] == 'BUY').any() else None,
            'sell_win_rate': float((t.loc[t['direction'] == 'SELL', 'net_return'] > 0).mean())
            if (t['direction'] == 'SELL').any() else None,
        }
    print(f"\n=== {name} ===")
    for seg_name in ('dev', 'holdout', 'all'):
        m = out.get(seg_name) or {}
        if m:
            print(f"  {seg_name:<8} sharpe={m['sharpe']:>6.2f}  "
                  f"ann={m['ann_return']:>+7.1%}  maxDD={m['max_drawdown']:>+7.1%}  "
                  f"days={m['n_days']}")
    ts = out.get('trade_stats')
    if ts:
        def pct(x):
            return f"{x:.1%}" if isinstance(x, float) else "n/a"
        print(f"  trades={ts['n_trades']}  win={pct(ts['win_rate'])}  "
              f"BUY win={pct(ts['buy_win_rate'])}  SELL win={pct(ts['sell_win_rate'])}")
    return out


def main(modes: list[str], horizon: int = 1, hold_days: int | None = None):
    """Simulate the panel for `horizon`, holding positions `hold_days`
    (defaults to the horizon itself so the trade expresses the prediction
    over its own window)."""
    hold_days = hold_days or max(horizon, 1)
    panel = _load_panel_with_execution_prices(horizon)
    print(f"panel (h={horizon}): {len(panel):,} rows, "
          f"{panel['date'].min().date()} .. {panel['date'].max().date()}")

    results = {'generated_at': datetime.now().isoformat(),
               'horizon': horizon, 'hold_days': hold_days,
               'costs': {'commission': COMMISSION, 'slippage': SLIPPAGE},
               'selection': {'top_n': TOP_N, 'min_stocks': MIN_STOCKS},
               'holdout_start': HOLDOUT_START}
    suffix = f"_h{horizon}"

    if 'legacy' in modes:
        daily, trades = simulate_legacy(panel)
        results['legacy'] = _report('legacy (1-day hold, next-open entry)',
                                    daily, trades)
        daily.rename('ret').to_csv(RESULTS_DIR / f"equity_legacy{suffix}.csv")

    if 'staggered' in modes:
        prices = pd.read_parquet(DATA_PATH, columns=['date', 'symbol', 'open', 'close'])
        sessions = np.sort(prices['date'].unique())
        wide = {
            'sessions': list(sessions),
            'open': prices.set_index(['date', 'symbol'])['open'].to_dict(),
            'close': prices.set_index(['date', 'symbol'])['close'].to_dict(),
        }
        daily, trades = simulate_staggered(panel, wide, hold_days=hold_days)
        results['staggered'] = _report(
            f'staggered ({hold_days}-day hold, {hold_days} tranches, '
            f'h={horizon} signal)', daily, trades)
        daily.rename('ret').to_csv(RESULTS_DIR / f"equity_staggered{suffix}.csv")

    with open(RESULTS_DIR / f"simulation_report{suffix}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved: {RESULTS_DIR / f'simulation_report{suffix}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["legacy", "staggered", "both"],
                    default="both")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 20],
                    help="Which walk-forward panel to simulate")
    ap.add_argument("--hold-days", type=int, default=None,
                    help="Holding period (defaults to horizon)")
    args = ap.parse_args()
    modes = ["legacy", "staggered"] if args.mode == "both" else [args.mode]
    main(modes, horizon=args.horizon, hold_days=args.hold_days)
