"""Factor-candidate pipeline (merged-B item 4).

Literature factors tested ONE AT A TIME as a 7th factor group through
the standard machinery: members-only universe with the point-in-time
mask (the post-PIT convention), h=20 walk-forward with tail fold,
four-tier verdict against the pitON baseline.

Candidates (all trailing-only, computed from OHLCV):
    amihud     Amihud (2002) illiquidity: mean(|ret| / dollar_vol, 20d)
    reversal   short-term reversal (Jegadeesh 1990): -ret_5d, -ret_21d
    high52     George & Hwang (2004): close / 252d rolling high
    seasonal   Heston & Sadka (2008) lite: mean same-calendar-month
               return over PRIOR years (expanding, strictly historical)

Multiple-testing discipline: every hypothesis this project has ever
adjudicated lives in hypothesis_ledger.csv; each new verdict prints the
family size N and the Sidak-adjusted t threshold for a 5% family-wise
error rate. A candidate that clears the per-test bar but not the
family bar is labeled accordingly -- no silent cherry-picking.
Rulebook v2 (2026-09-02, evaluation/rulebook.py): this script is TRACK A
(literature-backed, human-written candidates). Agent-mined candidates
are track B and never enter this family or this bar.

Usage:
    python evaluation/experiments/factor_pipeline.py --factor amihud --max-folds 1
    python evaluation/experiments/factor_pipeline.py          # all four
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from factors.factor_definitions import FACTOR_GROUPS  # noqa: E402
from features.cross_sectional import add_cross_sectional_zscore  # noqa: E402
from evaluation.purged_walk_forward import (  # noqa: E402
    WalkForwardConfig, run_walk_forward, DATA_PATH, RESULTS_DIR)
from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402
from evaluation.experiments.pit_universe_test import membership_mask  # noqa: E402
from evaluation.rulebook import family_size_literature, sidak_bar  # noqa: E402

MEMBERSHIP = Path(__file__).resolve().parent.parent.parent / "data" / "sp500_membership.parquet"
LEDGER = RESULTS_DIR / "hypothesis_ledger.csv"
HORIZON = 20
from evaluation.rulebook import SEGMENTS  # noqa: E402  (single definition)

# Every hypothesis adjudicated before this pipeline existed (retro-filled
# from the commit log). The family size for multiple-testing corrections
# counts ALL of these, not just the convenient ones.
RETRO_HYPOTHESES = [
    ("2026-07-20", "horizon_h5_vs_h1", "finding: signal peaks at h=5"),
    ("2026-07-20", "horizon_h20_vs_h1", "finding: costs amortize at h=20"),
    ("2026-07-20", "vol_ext_factor", "rejected: virgin-negative (regime luck)"),
    ("2026-07-21", "news_factor_h5", "passed h5, not adopted (prod is h20)"),
    ("2026-07-21", "news_factor_h20", "rejected: zero IC at prod horizon"),
    ("2026-07-30", "label_demarket", "rejected: redundant w/ per-date zscore"),
    ("2026-07-30", "label_debeta", "rejected: beta estimation noise"),
    ("2026-07-30", "label_volscale", "rejected: double-counts vol sizing"),
    ("2026-07-30", "ivol_position_sizing", "ADOPTED: holdout-robust"),
    ("2026-08-09", "pead_factor", "rejected: fresh tier voted no"),
    ("2026-08-10", "member_universe_filter", "ADOPTED: dev IC doubles"),
    ("2026-08-10", "optimizer_v1", "rejected: turnover mismatch"),
]


def _g(df):
    return df.groupby('symbol', group_keys=False)


def build_amihud(df):
    ret = _g(df)['close'].transform(lambda s: s.pct_change())
    dvol = (df['close'] * df['volume']).replace(0, np.nan)
    illiq = (ret.abs() / dvol) * 1e9
    df['amihud_20'] = illiq.groupby(df['symbol']).transform(
        lambda s: s.rolling(20, min_periods=10).mean())
    df['amihud_60'] = illiq.groupby(df['symbol']).transform(
        lambda s: s.rolling(60, min_periods=30).mean())
    return df, ['amihud_20', 'amihud_60']


def build_reversal(df):
    df['rev_5'] = -_g(df)['close'].transform(lambda s: s.pct_change(5))
    df['rev_21'] = -_g(df)['close'].transform(lambda s: s.pct_change(21))
    return df, ['rev_5', 'rev_21']


def build_high52(df):
    roll_max = _g(df)['close'].transform(
        lambda s: s.rolling(252, min_periods=126).max())
    df['pct_52w_high'] = df['close'] / roll_max
    df['days_from_high'] = 1.0 - df['pct_52w_high']
    return df, ['pct_52w_high', 'days_from_high']


def build_seasonal(df):
    """Mean same-calendar-month return over strictly PRIOR years."""
    d = df[['symbol', 'date', 'close']].copy()
    naive = d['date'].dt.tz_localize(None)
    d['ym'] = naive.dt.to_period('M')
    monthly = (d.sort_values(['symbol', 'date'])
                 .groupby(['symbol', 'ym'])['close'].last()
                 .groupby('symbol').pct_change()
                 .rename('mret').reset_index())
    monthly['month'] = monthly['ym'].dt.month
    monthly['year'] = monthly['ym'].dt.year
    monthly = monthly.sort_values(['symbol', 'month', 'year'])
    grp = monthly.groupby(['symbol', 'month'])['mret']
    monthly['seasonal'] = grp.transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean())
    df['_ym'] = naive.dt.to_period('M')
    df = df.merge(monthly[['symbol', 'ym', 'seasonal']],
                  left_on=['symbol', '_ym'], right_on=['symbol', 'ym'],
                  how='left').drop(columns=['_ym', 'ym'])
    df = df.rename(columns={'seasonal': 'seasonal_month'})
    return df, ['seasonal_month']


def _make_fundamental_builder(group: str):
    """Wave-2 candidates share one PIT feature build (EDGAR extract)."""
    def build(df):
        from features.fundamental_features import build_fundamental_features
        df, feats = build_fundamental_features(df)
        return df, feats[group]
    return build


CANDIDATES = {
    'amihud': build_amihud,
    'reversal': build_reversal,
    'high52': build_high52,
    'seasonal': build_seasonal,
    # wave 2 (EDGAR fundamentals, pre-registered as a batch 2026-08-10)
    'value': _make_fundamental_builder('value'),
    'profitability': _make_fundamental_builder('profitability'),
    'investment': _make_fundamental_builder('investment'),
    'accruals': _make_fundamental_builder('accruals'),
}


def family_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Sidak-adjusted two-sided t threshold (track A of the rulebook)."""
    return sidak_bar(n_tests, alpha)


def load_ledger() -> pd.DataFrame:
    if LEDGER.exists():
        return pd.read_csv(LEDGER)
    df = pd.DataFrame(RETRO_HYPOTHESES,
                      columns=['date', 'hypothesis', 'verdict'])
    df.to_csv(LEDGER, index=False)
    return df


def append_ledger(name: str, verdict: str):
    led = load_ledger()
    led = pd.concat([led, pd.DataFrame([{
        'date': date.today().isoformat(),
        'hypothesis': name, 'verdict': verdict}])], ignore_index=True)
    led.to_csv(LEDGER, index=False)
    return len(led)


def verdict(tag: str, factor_name: str):
    raw_col = f'future_return_{HORIZON}d'
    base = pd.read_parquet(RESULTS_DIR / "oos_predictions_h20_pitON.parquet")
    var = pd.read_parquet(RESULTS_DIR / f"oos_predictions_h20_{tag}.parquet")

    load_ledger()                      # creates the file on first use
    # Track A family = non-'finding' ledger rows (+ this candidate); the same
    # definition factor_card.py uses, so both print the same bar.
    n_family = family_size_literature() + 1
    thr = family_threshold(n_family)
    print(f"\n[multiple testing] track A hypothesis #{n_family} in this "
          f"project's literature family; family-wise 5% needs |t| >= {thr:.2f}")

    print(f"{'segment':<14}{'baseline':>20}{'with ' + factor_name:>20}"
          f"{factor_name + ' alone':>20}")
    print("-" * 76)
    tiers = {}
    for name, start, end in SEGMENTS:
        def seg(d):
            tz = d['date'].dt.tz
            m = pd.Series(True, index=d.index)
            if start:
                m &= d['date'] >= pd.Timestamp(start).tz_localize(tz)
            if end:
                m &= d['date'] < pd.Timestamp(end).tz_localize(tz)
            return d[m]
        b = summarize_ic(daily_rank_ic(seg(base), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        v = summarize_ic(daily_rank_ic(seg(var), 'pred', target_col=raw_col),
                         nw_lags=HORIZON - 1)
        f = summarize_ic(daily_rank_ic(seg(var), f'factor_{factor_name}',
                                       target_col=raw_col),
                         nw_lags=HORIZON - 1)
        tiers[name] = (b, v, f)

        def fmt(s):
            return (f"{s['ic_mean']:+.4f} (t={s['t_stat']:.2f})"
                    if s['n_days'] else "n/a")
        print(f"{name:<14}{fmt(b):>20}{fmt(v):>20}{fmt(f):>20}")

    rows = [{'segment': k, 'base_ic': t[0]['ic_mean'], 'with_ic': t[1]['ic_mean'],
             'alone_ic': t[2]['ic_mean'], 'alone_t': t[2]['t_stat']}
            for k, t in tiers.items()]
    out = RESULTS_DIR / f"factor_verdict_{factor_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")
    return tiers


def main(factors: list[str], max_folds: int | None):
    print("loading dataset (members + PIT mask, post-PIT convention)...")
    df = pd.read_parquet(DATA_PATH)
    members = set(pd.read_parquet(MEMBERSHIP)['symbol'].unique())
    df = df[df['symbol'].isin(members)].copy()
    mask = membership_mask(df)
    df = df[mask].copy()
    print(f"universe: {df['symbol'].nunique()} symbols, {len(df):,} PIT rows")

    for fname in factors:
        print(f"\n{'#' * 78}\n# CANDIDATE: {fname}\n{'#' * 78}")
        work = df.copy()
        work, feats = CANDIDATES[fname](work)
        work = add_cross_sectional_zscore(work, feats, suffix='_xs',
                                          winsorize_pct=0.01)
        cov = work[feats[0]].notna().mean()
        print(f"features: {feats} (+xs), coverage {cov:.1%}")
        groups = {k: list(v) for k, v in FACTOR_GROUPS.items()}
        groups[fname] = feats + [f + '_xs' for f in feats]

        run_walk_forward(
            WalkForwardConfig(horizon=HORIZON, min_tail_test=15),
            max_folds=max_folds, df=work, factor_groups=groups,
            tag=f"cand_{fname}")
        if not max_folds:
            tiers = verdict(f"cand_{fname}", fname)
            n = append_ledger(f"factor_{fname}", "pending-review")
            print(f"[ledger] recorded as hypothesis #{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", choices=list(CANDIDATES) + ["all"],
                    default="all")
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    main(list(CANDIDATES) if args.factor == "all" else [args.factor],
         max_folds=args.max_folds)
