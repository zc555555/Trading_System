"""Shared factor-ensemble training core (P1, 2026-07).

One implementation of "train the 6 factor ensembles" used by BOTH the
purged walk-forward evaluation and the production trainer, so the two can
never drift. Replaces the copy-pasted training blocks in
train_multi_factor_models.py / train_multi_horizon_models.py.

Honesty rules baked in:
- The inner train/validation boundary is PURGED: the last
  ``horizon + embargo_days`` dates of the training slice are discarded so no
  training label overlaps the validation period.
- Model and factor scores are per-date cross-sectional rank ICs
  (evaluation.metrics.daily_rank_ic), not pooled Pearson.
- Base-model blend weights are proportional to max(rank IC, 0); a model with
  negative IC gets zero weight, never a negative one.
- Factor blend weights use the same ic_sqrt algorithm as production
  (factors.factor_weighting.compute_ic_proportional with power=0.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from factors.factor_definitions import FACTOR_GROUPS  # noqa: E402
from factors.factor_weighting import compute_ic_proportional  # noqa: E402
from evaluation.metrics import daily_rank_ic  # noqa: E402

META_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
             'future_return', 'label']

MODEL_HYPERPARAMS = dict(n_estimators=100, max_depth=5, learning_rate=0.05)


def _make_models(seed: int):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    hp = MODEL_HYPERPARAMS
    return {
        'xgb': XGBRegressor(n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
                            learning_rate=hp['learning_rate'], subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                            verbosity=0),
        'lgb': LGBMRegressor(n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
                             learning_rate=hp['learning_rate'], subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                             verbose=-1),
        'cat': CatBoostRegressor(iterations=hp['n_estimators'], depth=hp['max_depth'],
                                 learning_rate=hp['learning_rate'], random_state=seed,
                                 verbose=0),
    }


def purged_inner_split(dates: np.ndarray, val_quantile: float = 0.85,
                       purge_days: int = 6):
    """Split sorted unique dates into (inner_train, inner_val) with a purge gap.

    The last ``purge_days`` dates of the inner-train block are dropped so that
    no training label (which reaches ``horizon`` days forward) overlaps the
    validation period. ``purge_days`` should be >= label horizon + embargo.
    """
    dates = np.sort(np.asarray(dates))
    split_idx = int(len(dates) * val_quantile)
    inner_train = dates[:max(split_idx - purge_days, 0)]
    inner_val = dates[split_idx:]
    return inner_train, inner_val


def train_factor_ensembles(
    df_train: pd.DataFrame,
    horizon: int = 1,
    embargo_days: int = 5,
    val_quantile: float = 0.85,
    target_col: str = 'future_return',
    factor_groups: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train one xgb+lgb+cat ensemble per factor with honest inner validation.

    Returns dict with:
        ensembles      {factor: {models, weights, feature_cols, model_ics, factor_ic}}
        factor_scores  {factor: inner-val mean daily rank IC}
        factor_weights {factor: ic_sqrt blend weight}
        inner_split    metadata
    """
    factor_groups = factor_groups or FACTOR_GROUPS
    df = df_train[df_train[target_col].notna()]

    inner_train_dates, inner_val_dates = purged_inner_split(
        df['date'].unique(), val_quantile, purge_days=horizon + embargo_days)
    train_mask = df['date'].isin(inner_train_dates)
    val_mask = df['date'].isin(inner_val_dates)
    df_val_meta = df.loc[val_mask, ['date', target_col]]

    if verbose:
        print(f"  inner-train: {len(inner_train_dates)} days "
              f"(..{pd.Timestamp(inner_train_dates[-1]).date()}), "
              f"purge {horizon + embargo_days}d, "
              f"inner-val: {len(inner_val_dates)} days "
              f"({pd.Timestamp(inner_val_dates[0]).date()}..)")

    y_train = df.loc[train_mask, target_col].values
    ensembles, factor_scores = {}, {}

    for factor_name, factor_features in factor_groups.items():
        feats = [f for f in factor_features if f in df.columns]
        if not feats:
            if verbose:
                print(f"  [SKIP] {factor_name}: no features present")
            continue

        X = df[feats].to_numpy(dtype=np.float32, na_value=0.0)
        X_tr, X_va = X[train_mask.to_numpy()], X[val_mask.to_numpy()]

        models, model_ics, val_preds = {}, {}, {}
        for name, model in _make_models(seed).items():
            model.fit(X_tr, y_train)
            pred = model.predict(X_va)
            ic = daily_rank_ic(
                df_val_meta.assign(_pred=pred), '_pred', target_col)
            models[name] = model
            model_ics[name] = float(ic.mean()) if len(ic) else np.nan
            val_preds[name] = pred

        # Base-model weights: proportional to positive rank IC only.
        clipped = {k: max(v, 0.0) for k, v in model_ics.items()
                   if np.isfinite(v)}
        total = sum(clipped.values())
        if total > 0:
            weights = {k: clipped.get(k, 0.0) / total for k in models}
        else:
            weights = {k: 1.0 / len(models) for k in models}

        blended = sum(val_preds[k] * weights[k] for k in models)
        factor_ic_series = daily_rank_ic(
            df_val_meta.assign(_pred=blended), '_pred', target_col)
        factor_ic = float(factor_ic_series.mean()) if len(factor_ic_series) else np.nan

        ensembles[factor_name] = {
            'models': models,
            'weights': weights,
            'feature_cols': feats,
            'model_ics': model_ics,
            'factor_ic': factor_ic,
        }
        factor_scores[factor_name] = factor_ic if np.isfinite(factor_ic) else 0.0
        if verbose:
            per_model = "  ".join(f"{k}={v:+.4f}" for k, v in model_ics.items())
            print(f"  [{factor_name:<10}] {per_model}  -> rankIC={factor_ic:+.4f}")

    # Factor blend weights: same ic_sqrt algorithm production uses.
    factor_weights = compute_ic_proportional(factor_scores, floor=0.0, power=0.5)
    if factor_weights is None:  # every factor <= 0: equal-weight fallback
        factor_weights = {k: 1.0 / len(factor_scores) for k in factor_scores}

    return {
        'ensembles': ensembles,
        'factor_scores': factor_scores,
        'factor_weights': factor_weights,
        'inner_split': {
            'val_quantile': val_quantile,
            'purge_days': horizon + embargo_days,
            'inner_train_end': str(pd.Timestamp(inner_train_dates[-1]).date()),
            'inner_val_start': str(pd.Timestamp(inner_val_dates[0]).date()),
        },
    }


def refit_on_full_data(fold: dict, df_train: pd.DataFrame,
                       target_col: str = 'future_return',
                       seed: int = 42, verbose: bool = True) -> dict:
    """Refit every base model on ALL labeled rows, keeping the weights and
    scores that were determined honestly on the purged inner validation.

    Production wants models trained on the freshest data; the inner split
    exists only to measure ICs and set weights. Weight determination and
    final fitting are therefore separated: measure first, then refit.
    """
    df = df_train[df_train[target_col].notna()]
    y = df[target_col].values
    refitted = {}
    for factor_name, ens in fold['ensembles'].items():
        X = df[ens['feature_cols']].to_numpy(dtype=np.float32, na_value=0.0)
        models = {}
        for name, model in _make_models(seed).items():
            model.fit(X, y)
            models[name] = model
        refitted[factor_name] = {**ens, 'models': models}
        if verbose:
            print(f"  [refit] {factor_name}: {len(df):,} rows, "
                  f"{len(ens['feature_cols'])} features")
    return {**fold, 'ensembles': refitted}


def _per_date_zscore(s: pd.Series, dates: pd.Series) -> pd.Series:
    """Z-score values within each date (mirrors production signal blending)."""
    g = s.groupby(dates.values)
    mean, std = g.transform('mean'), g.transform('std').replace(0, np.nan)
    return ((s - mean) / std).fillna(0.0)


def predict_panel(fold: dict, df: pd.DataFrame,
                  target_col: str = 'future_return') -> pd.DataFrame:
    """Score a frame with a trained fold: per-factor preds -> z-score -> blend.

    Returns [date, symbol, close, open, <target_col>, factor_*, pred].
    """
    keep = [c for c in ['date', 'symbol', 'open', 'close', target_col]
            if c in df.columns]
    out = df[keep].copy()

    blended_cols = []
    for factor_name, ens in fold['ensembles'].items():
        X = df[ens['feature_cols']].to_numpy(dtype=np.float32, na_value=0.0)
        raw = np.zeros(len(df))
        for name in sorted(ens['models']):
            raw += ens['models'][name].predict(X) * ens['weights'][name]
        col = f'factor_{factor_name}'
        # Per-date z-score so factors with different natural scales blend
        # fairly (identical to get_daily_signals_multi_factor._per_date_zscore)
        out[col] = _per_date_zscore(pd.Series(raw, index=df.index), df['date'])
        blended_cols.append((col, fold['factor_weights'].get(factor_name, 0.0)))

    out['pred'] = sum(out[c] * w for c, w in blended_cols)
    return out
