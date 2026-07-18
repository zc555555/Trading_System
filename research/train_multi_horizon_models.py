"""
MULTI-HORIZON FACTOR MODEL TRAINING (W2-A, 2026-05).

Trains one factor ensemble per (factor, horizon) pair on the existing
stocks_with_time_windows.parquet, using LOG returns at horizons
[1d, 5d, 20d] as labels. No changes to the upstream feature pipeline --
labels are computed on-the-fly here.

Why
---
The 1-day horizon is the bottleneck for trend/volume/market factors.
Cross-sectional momentum, mean-vol anomalies, and volume bursts predict
multi-day moves, not next-day moves. This experiment measures IC for
every (factor, horizon) cell so we can pick the right horizon per factor.

Output
------
artifacts/multi_horizon/
    ensemble_{horizon}d_{factor}.pkl    -- one ensemble per cell (18 files)
    multi_horizon_metrics.json          -- IC matrix + diagnostics
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from factors.factor_definitions import FACTOR_GROUPS, verify_feature_coverage


HORIZONS = [1, 5, 20]
PARQUET = Path(__file__).parent / "data" / "stocks_with_time_windows.parquet"
OUT_DIR = Path(__file__).parent / "artifacts" / "multi_horizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_horizon_labels(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add future_return_{h}d columns (log returns) for each horizon.

    Drops rows where the longest-horizon label is NaN so all horizons share
    the same training set -- this makes IC comparable across horizons.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for h in horizons:
        col = f"future_return_{h}d"
        df[col] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x.shift(-h) / x)
        )
    longest = max(horizons)
    before = len(df)
    df = df.dropna(subset=[f"future_return_{longest}d"]).reset_index(drop=True)
    print(f"[labels] Computed horizons {horizons}; dropped "
          f"{before - len(df):,} rows missing the {longest}d label "
          f"(common train set = {len(df):,} rows)")
    return df


def train_ensemble(X_train, y_train, X_val, y_val, tag: str):
    """Train XGB+LGB+CAT ensemble. Returns dict with models, weights, IC."""
    models = {}

    xgb = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)
    xgb_ic = float(np.corrcoef(xgb_pred, y_val)[0, 1])
    models["xgb"] = xgb

    lgb = LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_val)
    lgb_ic = float(np.corrcoef(lgb_pred, y_val)[0, 1])
    models["lgb"] = lgb

    cat = CatBoostRegressor(
        iterations=100, depth=5, learning_rate=0.05,
        random_state=42, verbose=0,
    )
    cat.fit(X_train, y_train)
    cat_pred = cat.predict(X_val)
    cat_ic = float(np.corrcoef(cat_pred, y_val)[0, 1])
    models["cat"] = cat

    scores = {"xgb": xgb_ic, "lgb": lgb_ic, "cat": cat_ic}
    total = sum(scores.values())
    if total > 0:
        weights = {k: v / total for k, v in scores.items()}
    else:
        weights = {"xgb": 0.34, "lgb": 0.33, "cat": 0.33}

    ensemble_pred = (
        xgb_pred * weights["xgb"]
        + lgb_pred * weights["lgb"]
        + cat_pred * weights["cat"]
    )
    ensemble_ic = float(np.corrcoef(ensemble_pred, y_val)[0, 1])

    print(f"  [{tag}] xgb={xgb_ic:+.4f}  lgb={lgb_ic:+.4f}  cat={cat_ic:+.4f}  "
          f"ensemble={ensemble_ic:+.4f}")

    return {
        "models": models,
        "weights": weights,
        "val_score": ensemble_ic,
        "individual_scores": scores,
    }


def main():
    print("=" * 80)
    print("MULTI-HORIZON FACTOR MODEL TRAINING (W2-A)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----- load + label -----
    print(f"\n[1/4] Loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET)
    print(f"      Shape: {df.shape}, "
          f"date range {df['date'].min().date()} -> {df['date'].max().date()}, "
          f"{df['symbol'].nunique()} symbols")

    # Use only rows that already have ANY label (downstream feature build
    # dropped trailing days). Re-compute multi-horizon labels.
    print(f"\n[2/4] Computing labels for horizons {HORIZONS} ...")
    df = compute_horizon_labels(df, HORIZONS)

    # ----- factor coverage check -----
    print(f"\n[3/4] Verifying factor feature coverage ...")
    cov = verify_feature_coverage(df.columns.tolist())
    for factor_name, info in cov.items():
        status = "[OK]" if info["missing"] == 0 else "[WARN]"
        print(f"  {status} {factor_name}: {info['available']}/{info['total']}"
              f" ({info['coverage_pct']:.0f}%)")
        if info["missing"] > 0:
            print(f"       missing: {info['missing_features']}")

    # ----- 80/20 time split (same as training script) -----
    split_date = df["date"].quantile(0.8)
    train_mask = df["date"] < split_date
    val_mask = df["date"] >= split_date
    print(f"\nTrain: {train_mask.sum():,} samples (< {split_date.date()})")
    print(f"Val  : {val_mask.sum():,} samples (>= {split_date.date()})")

    # ----- train one ensemble per (factor, horizon) -----
    print(f"\n[4/4] Training 6 factors x {len(HORIZONS)} horizons = "
          f"{6*len(HORIZONS)} ensembles ...")
    print(f"      (3 base models per ensemble = "
          f"{6*len(HORIZONS)*3} model fits total)")

    results = {}  # results[horizon][factor] = {ic, weights, feature_cols, ...}

    for h in HORIZONS:
        y_col = f"future_return_{h}d"
        y_train = df.loc[train_mask, y_col].values
        y_val = df.loc[val_mask, y_col].values
        results[h] = {}

        print(f"\n--- Horizon: {h}d ---")
        print(f"  Label stats: train mean={y_train.mean():+.4f} "
              f"std={y_train.std():.4f}, val mean={y_val.mean():+.4f} "
              f"std={y_val.std():.4f}")

        for factor_name, factor_features in FACTOR_GROUPS.items():
            available = [f for f in factor_features if f in df.columns]
            if not available:
                print(f"  [{h}d/{factor_name}] no features available, skipping")
                continue

            X = df[available].values
            X = pd.DataFrame(X, columns=available).ffill().fillna(0).values
            X_train = X[train_mask.values]
            X_val = X[val_mask.values]

            ens = train_ensemble(
                X_train, y_train, X_val, y_val,
                tag=f"{h}d/{factor_name}",
            )

            results[h][factor_name] = {
                "models": ens["models"],
                "weights": ens["weights"],
                "val_score": ens["val_score"],
                "individual_scores": ens["individual_scores"],
                "feature_cols": available,
                "horizon": h,
            }

            # save model
            out_path = OUT_DIR / f"ensemble_{h}d_{factor_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(results[h][factor_name], f)

    # ----- summary matrix -----
    print(f"\n{'=' * 80}\nIC MATRIX (validation correlation)\n{'=' * 80}")
    factor_names = sorted({f for h in results for f in results[h]})
    header = f"  {'factor':<12} " + "  ".join(f"{h}d".rjust(10) for h in HORIZONS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    ic_matrix = {}
    for f in factor_names:
        row = []
        ic_matrix[f] = {}
        for h in HORIZONS:
            ic = results[h].get(f, {}).get("val_score", float("nan"))
            ic_matrix[f][h] = ic
            row.append(f"{ic:+.4f}".rjust(10))
        print(f"  {f:<12} " + "  ".join(row))

    # ----- best horizon per factor -----
    print(f"\n{'=' * 80}\nBEST HORIZON PER FACTOR\n{'=' * 80}")
    best_horizon = {}
    for f in factor_names:
        ics = {h: ic_matrix[f][h] for h in HORIZONS if not np.isnan(ic_matrix[f][h])}
        if not ics:
            continue
        best_h = max(ics, key=ics.get)
        best_ic = ics[best_h]
        cur_ic = ics.get(1, float("nan"))
        gain = best_ic - cur_ic if not np.isnan(cur_ic) else float("nan")
        marker = "*" if best_h != 1 and gain > 0.01 else " "
        print(f"  {marker} {f:<12} best={best_h}d (IC={best_ic:+.4f})  "
              f"vs 1d (IC={cur_ic:+.4f})  gain={gain:+.4f}")
        best_horizon[f] = {"horizon": best_h, "ic": best_ic, "gain_over_1d": gain}

    # ----- persist metrics -----
    metrics_path = OUT_DIR / "multi_horizon_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now().isoformat(),
            "horizons": HORIZONS,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "split_date": str(split_date.date()),
            "ic_matrix": {
                fname: {str(h): ic for h, ic in cells.items()}
                for fname, cells in ic_matrix.items()
            },
            "best_horizon_per_factor": {
                k: {"horizon": v["horizon"], "ic": v["ic"], "gain_over_1d": v["gain_over_1d"]}
                for k, v in best_horizon.items()
            },
            "feature_counts": {
                f: len(results[1].get(f, {}).get("feature_cols", []))
                for f in factor_names
            },
        }, f, indent=2)
    print(f"\n[OK] Metrics saved -> {metrics_path}")

    print(f"\n{'=' * 80}\nTRAINING COMPLETE\n{'=' * 80}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
