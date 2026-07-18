"""
Evaluate W2-A multi-horizon signals at MULTIPLE return horizons.

The default win-rate inside get_daily_signals_multi_factor.py compares
predictions against future_return (1d). After W2-A promoted long-horizon
models, that's apples-to-oranges -- a 20d-trained momentum model should
be judged against 20d returns, not 1d.

This script:
  1. Loads the production factor models (post-W2A promotion).
  2. Predicts on the last ~500 labeled rows, blends with per-date z-score
     + ic_sqrt weights.
  3. Evaluates win rate at each of 1d / 5d / 20d future-return horizons.
  4. Reports the win rate where the system is actually designed to work.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from factors.factor_definitions import FACTOR_GROUPS
from factors.factor_weighting import load_effective_factor_weights


def main():
    ARTIFACTS = Path(__file__).parent / "artifacts"
    DATA = Path(__file__).parent / "data" / "stocks_with_time_windows.parquet"

    df = pd.read_parquet(DATA)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"Loaded {df.shape}, {df['symbol'].nunique()} symbols")

    # Compute future returns at 1d / 5d / 20d for evaluation
    for h in [1, 5, 20]:
        df[f"fr_{h}d"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x.shift(-h) / x)
        )

    # Keep last 60 unique dates with valid 20d label (need full cross-section
    # per date so z-scoring works properly).
    valid = df.dropna(subset=["fr_20d"])
    last_dates = sorted(valid["date"].unique())[-60:]
    df_eval = valid[valid["date"].isin(last_dates)].copy().reset_index(drop=True)
    print(f"Evaluation set: {len(df_eval):,} rows across "
          f"{df_eval['date'].nunique()} dates")

    # Load production models + weights
    FACTOR_WEIGHTS = load_effective_factor_weights(log=False)
    print(f"\nEffective weights: {FACTOR_WEIGHTS}")

    factor_predictions = {}
    for fname in FACTOR_GROUPS:
        pkl = ARTIFACTS / f"ensemble_{fname}.pkl"
        if not pkl.exists():
            print(f"  [skip] {fname}: no model")
            continue
        ens = pickle.load(open(pkl, "rb"))
        feats = [c for c in ens["feature_cols"] if c in df_eval.columns]
        X = pd.DataFrame(df_eval[feats].values, columns=feats).ffill().fillna(0).values

        # ensemble prediction
        sub_models = ens["models"]
        sub_weights = ens["weights"]
        preds_by_model = {
            m: sub_models[m].predict(X) for m in sorted(sub_models.keys())
        }
        stacked = np.column_stack([preds_by_model[m] for m in sorted(preds_by_model.keys())])
        sub_w = np.array([sub_weights[m] for m in sorted(preds_by_model.keys())])
        factor_pred = stacked @ sub_w

        # per-date z-score
        tmp = pd.DataFrame({"date": df_eval["date"].values, "p": factor_pred})
        tmp["pz"] = tmp.groupby("date")["p"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )
        factor_predictions[fname] = tmp["pz"].values

    # Blend
    blended = np.zeros(len(df_eval))
    for fname, pred in factor_predictions.items():
        w = FACTOR_WEIGHTS.get(fname, 0.0)
        blended += pred * w

    # IC + win rate per horizon
    print(f"\n{'='*70}")
    print("EVALUATION ACROSS HORIZONS")
    print(f"{'='*70}")
    print(f"{'horizon':<10}{'IC':<12}{'overall WR':<13}{'BUY WR':<10}{'SELL WR':<10}{'BUY/SELL n':<12}")
    print("-" * 70)
    for h in [1, 5, 20]:
        y = df_eval[f"fr_{h}d"].values
        mask = np.isfinite(y)
        b = blended[mask]
        y2 = y[mask]
        ic = float(np.corrcoef(b, y2)[0, 1]) if b.std() > 0 else float("nan")
        buy_mask = b > 0
        sell_mask = b < 0
        overall_wr = ((b > 0) == (y2 > 0)).mean()
        buy_wr = ((b[buy_mask] > 0) == (y2[buy_mask] > 0)).mean() if buy_mask.any() else float("nan")
        sell_wr = ((b[sell_mask] < 0) == (y2[sell_mask] < 0)).mean() if sell_mask.any() else float("nan")
        print(f"  {h}d{'':<7}{ic:>+8.4f}    {overall_wr*100:>6.2f}%      "
              f"{buy_wr*100:>5.1f}%    {sell_wr*100:>5.1f}%   "
              f"{int(buy_mask.sum())}/{int(sell_mask.sum())}")

    # Top-decile evaluation (the real strategy: buy top decile, sell bottom decile)
    print(f"\n{'='*70}")
    print("TOP/BOTTOM DECILE PORTFOLIO (per-date)")
    print(f"{'='*70}")
    df_eval["blended"] = blended
    for h in [1, 5, 20]:
        y_col = f"fr_{h}d"
        if df_eval[y_col].isna().all():
            continue
        # rank within each date
        df_eval["rank_pct"] = df_eval.groupby("date")["blended"].rank(pct=True)
        top = df_eval[df_eval["rank_pct"] >= 0.9]
        bot = df_eval[df_eval["rank_pct"] <= 0.1]
        top_ret = top[y_col].mean()
        bot_ret = bot[y_col].mean()
        ls = top_ret - bot_ret
        print(f"  {h}d:  top10% avg={top_ret*100:+.3f}%   "
              f"bot10% avg={bot_ret*100:+.3f}%   "
              f"long-short={ls*100:+.3f}%   "
              f"top WR={((top[y_col] > 0).mean()*100):.1f}%   "
              f"bot WR={((bot[y_col] < 0).mean()*100):.1f}%")


if __name__ == "__main__":
    main()
