"""Factor report card -- the four checks IC alone does not give.

For every factor score column in a walk-forward prediction panel (per-date
scores + matured label), report:

  1. IC by evidence tier            (what we already had)
  2. Decile analysis                mean label per score decile, top-minus-
                                    bottom spread with NW t, monotonicity
                                    (Spearman of decile index vs mean return)
  3. Turnover and capacity          top-decile membership overlap across one
                                    holding period -> implied turnover; the
                                    spread net of round-trip costs
  4. Orthogonality                  correlation matrix of per-date z-scored
                                    factor scores, and RESIDUAL IC: the
                                    factor orthogonalised cross-sectionally
                                    against all other factors each date --
                                    the IC of its genuinely new information
  5. Gate                           machine verdict under the ledger's
                                    family-wise bar (current rules); printed
                                    with the rule version so a future rule
                                    change is visible in the record

A factor that scores well on 1 and badly on 4 is a relabelled copy of an
existing factor; one that scores well on 1-2 and badly on 3 is a cost
sink. Descriptive only; nothing here changes weights.

Usage:
    python evaluation/factor_card.py --panel surv
    python evaluation/factor_card.py --panel surv --factors factor_high52,factor_momentum
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import daily_rank_ic, summarize_ic  # noqa: E402
from evaluation.rulebook import (  # noqa: E402
    RULE_VERSION, SEGMENTS as _SEGMENTS, family_size_literature, gate_literature, sidak_bar)

RESULTS = Path(__file__).resolve().parent / "results"
IMG = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
HORIZON = 20
N_DECILES = 10
MIN_NAMES = 40
ROUND_TRIP_BP = 15.0
SEGMENTS = _SEGMENTS                     # single definition: evaluation/rulebook.py
# RULE_VERSION now lives in evaluation/rulebook.py (v2, two-track).


# --------------------------------------------------------------------------
# pure functions
# --------------------------------------------------------------------------
def _seg(df: pd.DataFrame, start, end) -> pd.DataFrame:
    d = df["date"]
    tz = getattr(d.dt, "tz", None)
    lo = pd.Timestamp(start).tz_localize(tz) if (start and tz) else (pd.Timestamp(start) if start else None)
    hi = pd.Timestamp(end).tz_localize(tz) if (end and tz) else (pd.Timestamp(end) if end else None)
    m = pd.Series(True, index=df.index)
    if lo is not None:
        m &= d >= lo
    if hi is not None:
        m &= d < hi
    return df[m]


def per_date_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("date")[cols]
    return (df[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def decile_table(df: pd.DataFrame, col: str, label: str,
                 n_dec: int = N_DECILES, min_names: int = MIN_NAMES) -> dict:
    """Mean label per score decile (averaged over dates), spread, monotonicity."""
    sub = df[["date", col, label]].dropna()
    counts = sub.groupby("date")[col].transform("count")
    sub = sub[counts >= min_names]
    if sub.empty:
        return {"n_dates": 0}
    ranks = sub.groupby("date")[col].rank(pct=True, method="first")
    sub = sub.assign(dec=np.minimum((ranks * n_dec).astype(int) + 1, n_dec))
    by_date = sub.groupby(["date", "dec"])[label].mean().unstack("dec")
    means = by_date.mean()
    spread = by_date[n_dec] - by_date[1]
    s = summarize_ic(spread, nw_lags=HORIZON - 1)
    mono = pd.Series(means.values).corr(pd.Series(np.arange(1, n_dec + 1)), method="spearman")
    return {"n_dates": int(len(by_date)),
            "decile_means": [float(x) for x in means.values],
            "spread_mean": float(spread.mean()),
            "spread_t": float(s.get("t_stat", np.nan)),
            "monotonicity": float(mono),
            "share_dates_positive_spread": float((spread > 0).mean())}


def turnover(df: pd.DataFrame, col: str, hold: int = HORIZON, top_frac: float = 0.1) -> dict:
    """Top-fraction membership overlap between t and t+hold sessions."""
    sub = df[["date", "symbol", col]].dropna()
    dates = np.sort(sub["date"].unique())
    if len(dates) <= hold:
        return {"turnover": np.nan}
    top = {}
    for d, g in sub.groupby("date"):
        k = max(1, int(len(g) * top_frac))
        top[d] = set(g.nlargest(k, col)["symbol"])
    tos = []
    for i in range(len(dates) - hold):
        a, b = top.get(dates[i]), top.get(dates[i + hold])
        if a and b:
            tos.append(1.0 - len(a & b) / max(len(a), 1))
    to = float(np.mean(tos)) if tos else np.nan
    ac = (sub.sort_values(["symbol", "date"]).groupby("symbol")[col]
          .apply(lambda s: s.autocorr(lag=hold) if len(s) > hold + 5 else np.nan).mean())
    return {"turnover_per_hold": to, "score_autocorr_lag_h": float(ac)}


def residual_ic(df: pd.DataFrame, col: str, others: list[str], label: str) -> pd.Series:
    """Per-date IC of `col` after cross-sectional OLS on `others`."""
    cols = [col] + others + [label]
    sub = df[["date"] + cols].dropna()
    out = {}
    for d, g in sub.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        X = np.column_stack([np.ones(len(g)), g[others].to_numpy()])
        y = g[col].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        out[d] = pd.Series(res).corr(pd.Series(g[label].to_numpy()), method="spearman")
    return pd.Series(out)


def gate(ic_by_tier: dict, family_bar: float) -> dict:
    """Track-A (literature) gate of the rulebook: holdout |t| >= family bar,
    and no tier significantly negative (t <= -2); fresh vetoes only with
    >= 60 sessions. Production factors are literature-backed, so the card
    reports them under track A. See evaluation/rulebook.py."""
    return gate_literature(ic_by_tier, family_bar)


# --------------------------------------------------------------------------
def family_bar_from_ledger() -> tuple[int, float]:
    """Track-A family size (non-'finding' ledger rows) and its Sidak bar."""
    n = family_size_literature()
    return n, sidak_bar(n)


def run(panel_path: Path, factors: list[str] | None, tag: str) -> dict:
    df = pd.read_parquet(panel_path)
    label = f"future_return_{HORIZON}d"
    fac = factors or [c for c in df.columns if c.startswith("factor_")
                      and df[c].notna().any() and df[c].nunique() > 1]
    df = df.dropna(subset=[label])
    z = per_date_zscore(df, fac)
    for c in fac:
        df[c] = z[c]
    n_hyp, bar = family_bar_from_ledger()

    cards = {}
    for c in fac:
        others = [o for o in fac if o != c]
        tiers = {}
        for name, s, e in SEGMENTS:
            seg = _seg(df, s, e)
            if seg.empty:
                continue
            ic = daily_rank_ic(seg, c, target_col=label)
            tiers[name] = summarize_ic(ic, nw_lags=HORIZON - 1)
        dev = _seg(df, "2021-12-30", "2025-07-01")
        hold = _seg(df, "2025-07-01", None)
        r_dev = residual_ic(dev, c, others, label)
        r_hold = residual_ic(hold, c, others, label)
        cards[c] = {
            "ic_by_tier": tiers,
            "deciles_dev": decile_table(dev, c, label),
            "deciles_holdout": decile_table(hold, c, label),
            "turnover": turnover(dev, c),
            "residual_ic_dev": summarize_ic(r_dev, nw_lags=HORIZON - 1),
            "residual_ic_holdout": summarize_ic(r_hold, nw_lags=HORIZON - 1),
            "gate": gate(tiers, bar),
        }
        # cost-adjusted spread: top-bottom spread per hold, minus the cost of
        # turning over both legs at the round trip
        dd = cards[c]["deciles_dev"]
        to = cards[c]["turnover"].get("turnover_per_hold", np.nan)
        if dd.get("n_dates") and np.isfinite(to):
            dd["spread_net_of_costs"] = float(dd["spread_mean"] - 2 * to * ROUND_TRIP_BP / 1e4)

    corr = z[fac].corr().round(3)
    out = {"generated_at": datetime.now().isoformat(), "panel": str(panel_path),
           "n_hypotheses_in_family": n_hyp, "family_bar_t": round(bar, 3),
           "factor_corr": corr.to_dict(), "cards": cards}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"factor_card_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    _print(out, fac)
    _chart(out, fac, IMG / f"factor_card_{tag}.png")
    return out


def _print(out: dict, fac: list[str]) -> None:
    print(f"\nfamily: N={out['n_hypotheses_in_family']} hypotheses -> Sidak bar |t| >= {out['family_bar_t']}")
    print(f"\n{'factor':<12}{'IC dev':>8}{'IC hold':>9}{'resid dev':>10}{'resid hold':>11}"
          f"{'spread dev':>11}{'mono':>6}{'turnover':>9}{'net spread':>11}{'gate':>6}")
    print("-" * 95)
    for c in fac:
        k = out["cards"][c]
        t = k["ic_by_tier"]
        dd = k["deciles_dev"]
        print(f"{c.replace('factor_', ''):<12}"
              f"{t.get('seen_dev', {}).get('ic_mean', np.nan):>8.4f}"
              f"{t.get('holdout', {}).get('ic_mean', np.nan):>9.4f}"
              f"{k['residual_ic_dev'].get('ic_mean', np.nan):>10.4f}"
              f"{k['residual_ic_holdout'].get('ic_mean', np.nan):>11.4f}"
              f"{dd.get('spread_mean', np.nan):>11.4f}"
              f"{dd.get('monotonicity', np.nan):>6.2f}"
              f"{k['turnover'].get('turnover_per_hold', np.nan):>9.2f}"
              f"{dd.get('spread_net_of_costs', np.nan):>11.4f}"
              f"{k['gate']['verdict']:>6}")
    print("\nfactor score correlation (per-date z-scores):")
    print(pd.DataFrame(out["factor_corr"]).rename(index=lambda s: s.replace("factor_", ""),
                                                  columns=lambda s: s.replace("factor_", "")).to_string())


def _chart(out: dict, fac: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(fac)
    fig, axes = plt.subplots(2, max(n, 1), figsize=(2.6 * max(n, 1), 6))
    axes = np.atleast_2d(axes)
    for j, c in enumerate(fac):
        for i, key in enumerate(("deciles_dev", "deciles_holdout")):
            d = out["cards"][c][key]
            ax = axes[i, j]
            if d.get("n_dates"):
                ax.bar(range(1, N_DECILES + 1), np.array(d["decile_means"]) * 100,
                       color="#4878CF")
                ax.set_title(f"{c.replace('factor_', '')} {'dev' if i == 0 else 'holdout'}\n"
                             f"spread {d['spread_mean']*100:+.2f}% mono {d['monotonicity']:.2f}",
                             fontsize=8)
            ax.axhline(0, color="black", lw=0.5)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"mean {HORIZON}d return (%)", fontsize=8)
    fig.suptitle("Decile analysis per factor (per-date z-scored scores; bars = mean forward return by decile)",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="surv", help="prediction panel tag (oos_predictions_h20_{tag})")
    ap.add_argument("--factors", default=None, help="comma-separated factor columns (default: all)")
    a = ap.parse_args()
    run(RESULTS / f"oos_predictions_h{HORIZON}_{a.panel}.parquet",
        a.factors.split(",") if a.factors else None, a.panel)
