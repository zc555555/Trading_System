"""Evaluation metrics: per-date cross-sectional rank IC and portfolio stats.

Why per-date rank IC instead of pooled Pearson
----------------------------------------------
The old pipeline scored factors with ``np.corrcoef(pred, y)`` pooled over all
validation rows. That number mixes cross-sectional skill (can we rank stocks
within a day?) with time-series co-movement (do predictions and returns drift
together across months?), and with overlapping multi-day labels it is inflated
by autocorrelation. The strategy trades cross-sectionally -- it buys the top
names each day -- so the honest question is: on each single day, how well do
predictions rank that day's returns? That is the per-date Spearman IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "future_return",
    date_col: str = "date",
    min_names_per_date: int = 20,
) -> pd.Series:
    """Per-date cross-sectional Spearman correlation of pred vs target.

    Dates with fewer than ``min_names_per_date`` valid (pred, target) pairs
    are skipped -- a rank correlation over a handful of names is noise.

    Returns a Series indexed by date.
    """
    sub = df[[date_col, pred_col, target_col]].dropna()
    out = {}
    for date, g in sub.groupby(date_col, sort=True):
        if len(g) < min_names_per_date:
            continue
        # Spearman = Pearson on ranks; rank() handles ties by averaging.
        pr = g[pred_col].rank()
        tr = g[target_col].rank()
        pr_std, tr_std = pr.std(), tr.std()
        if pr_std == 0 or tr_std == 0:
            continue
        out[date] = float(np.corrcoef(pr, tr)[0, 1])
    return pd.Series(out, name=f"ic_{pred_col}").sort_index()


def summarize_ic(ic: pd.Series) -> dict:
    """Mean IC, t-stat, and IC information ratio for a per-date IC series.

    t-stat treats daily ICs as i.i.d. -- with 1-day labels there is no
    mechanical overlap, so this is a reasonable first-order statistic.
    |t| >= 2 is the usual bar for "probably not noise".
    """
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"n_days": 0, "ic_mean": np.nan, "ic_std": np.nan,
                "t_stat": np.nan, "ic_ir": np.nan, "pct_positive": np.nan}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    t_stat = mean / std * np.sqrt(n) if std > 0 else np.nan
    return {
        "n_days": int(n),
        "ic_mean": mean,
        "ic_std": std,
        "t_stat": float(t_stat),
        "ic_ir": mean / std if std > 0 else np.nan,
        "pct_positive": float((ic > 0).mean()),
    }


def portfolio_metrics(daily_returns: pd.Series, trading_days: int = 252) -> dict:
    """Standard performance stats from a daily portfolio return series."""
    r = daily_returns.dropna()
    if len(r) == 0:
        return {}
    equity = (1 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = len(r) / trading_days
    ann_return = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(trading_days)) if r.std() > 0 else np.nan
    downside = r[r < 0]
    sortino = (float(r.mean() / downside.std(ddof=1) * np.sqrt(trading_days))
               if len(downside) > 1 and downside.std() > 0 else np.nan)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())
    return {
        "n_days": int(len(r)),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": ann_return / abs(max_dd) if max_dd < 0 else np.nan,
        "daily_win_rate": float((r > 0).mean()),
    }
