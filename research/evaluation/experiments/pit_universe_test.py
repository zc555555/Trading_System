"""PIT-universe contrast experiment (merged-B item 2).

Question: how much of our measured IC is an artifact of membership
look-ahead -- evaluating past dates on TODAY'S index members, thereby
riding the pre-inclusion runup of stocks that were added later?

Design: restrict to symbols that appear in the S&P membership table at
all (drops ETFs and never-members so both runs share one universe),
then run the h=20 walk-forward twice:

    pitOFF  today's-members convention (the incumbent methodology)
    pitON   each (date, symbol) row kept only if the symbol was an
            index member ON THAT DATE (sp500_membership.parquet)

Both runs use the tail fold. The four-tier blended-IC comparison
quantifies the bias. Residual (documented): delisted members without
price history cannot be added back; pitON removes look-ahead inclusions
but the dead are still missing from both runs.

Usage:
    python evaluation/experiments/pit_universe_test.py --max-folds 1
    python evaluation/experiments/pit_universe_test.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from evaluation.purged_walk_forward import (  # noqa: E402
    WalkForwardConfig, run_walk_forward, DATA_PATH, RESULTS_DIR)
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402

MEMBERSHIP = Path(__file__).resolve().parent.parent.parent / "data" / "sp500_membership.parquet"
HORIZON = 20
SEGMENTS = [
    ("virgin_early", None, "2021-12-30"),
    ("seen_dev", "2021-12-30", "2025-07-01"),
    ("holdout", "2025-07-01", "2026-05-16"),
    ("fresh", "2026-05-16", None),
]


def membership_mask(df: pd.DataFrame) -> pd.Series:
    """True where the row's symbol was an index member on the row's date."""
    iv = pd.read_parquet(MEMBERSHIP)
    naive = df['date'].dt.tz_localize(None)
    mask = pd.Series(False, index=df.index)
    for sym, g in iv.groupby('symbol'):
        sel = df['symbol'] == sym
        if not sel.any():
            continue
        d = naive[sel]
        m = pd.Series(False, index=d.index)
        for _, r in g.iterrows():
            hi = r['end'] if pd.notna(r['end']) else pd.Timestamp('2100-01-01')
            m |= (d >= r['start']) & (d < hi)
        mask.loc[d.index] = m
    return mask


def compare(tag_off: str, tag_on: str):
    raw_col = f'future_return_{HORIZON}d'
    off = pd.read_parquet(RESULTS_DIR / f"oos_predictions_h20_{tag_off}.parquet")
    on = pd.read_parquet(RESULTS_DIR / f"oos_predictions_h20_{tag_on}.parquet")

    print(f"\n{'segment':<14}{'today-members (pitOFF)':>26}{'point-in-time (pitON)':>26}")
    print("-" * 68)
    rows = []
    for name, start, end in SEGMENTS:
        def seg(d):
            tz = d['date'].dt.tz
            m = pd.Series(True, index=d.index)
            if start:
                m &= d['date'] >= pd.Timestamp(start).tz_localize(tz)
            if end:
                m &= d['date'] < pd.Timestamp(end).tz_localize(tz)
            return d[m]
        a = summarize_ic(daily_rank_ic(seg(off), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        b = summarize_ic(daily_rank_ic(seg(on), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)

        def fmt(s):
            return (f"{s['ic_mean']:+.4f} (t={s['t_stat']:.2f}, n={s['n_days']})"
                    if s['n_days'] else "n/a")
        print(f"{name:<14}{fmt(a):>26}{fmt(b):>26}")
        rows.append({"segment": name, "pitOFF_ic": a['ic_mean'],
                     "pitON_ic": b['ic_mean'], "pitOFF_t": a['t_stat'],
                     "pitON_t": b['t_stat']})
    out = RESULTS_DIR / "pit_universe_verdict.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


def main(max_folds: int | None):
    print("loading dataset + membership...")
    df = pd.read_parquet(DATA_PATH)
    iv = pd.read_parquet(MEMBERSHIP)
    sp_symbols = set(iv['symbol'].unique())
    subset = df[df['symbol'].isin(sp_symbols)].copy()
    dropped = sorted(set(df['symbol'].unique()) - sp_symbols)
    print(f"S&P-member subset: {subset['symbol'].nunique()} symbols "
          f"(dropped {len(dropped)} never-members/ETFs: {dropped[:8]}...)")

    mask = membership_mask(subset)
    print(f"PIT filter keeps {mask.mean():.1%} of subset rows")

    for tag, data in (("pitOFF", subset), ("pitON", subset[mask].copy())):
        run_walk_forward(
            WalkForwardConfig(horizon=HORIZON, min_tail_test=15),
            max_folds=max_folds,
            df=data,
            tag=tag,
        )
    if not max_folds:
        compare("pitOFF", "pitON")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    main(max_folds=args.max_folds)
