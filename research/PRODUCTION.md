# Production Assets — What's Live

This file is the single source of truth for "which scripts and models are actually in the daily/weekly production pipeline." Anything not listed here is either upstream tooling (data/features) or archived under `legacy/` / `artifacts/deprecated/`.

---

## Production entry points

| Script | Cadence | Called by | Purpose |
|---|---|---|---|
| [data/fetch_ohlcv.py](data/fetch_ohlcv.py) | daily / weekly | `run_auto_trading.py`, `run_retrain_models.py` | Pull latest OHLCV → `data/stocks.parquet` |
| [prepare_prediction_data.py](prepare_prediction_data.py) | daily | `run_auto_trading.py` | Rebuild features → `data/stocks_selected_features.parquet` |
| [get_daily_signals_multi_factor.py](get_daily_signals_multi_factor.py) | daily | `run_auto_trading.py`, `run_menu.py` | Load 6 factor ensembles → emit `artifacts/signals_multi_factor_YYYYMMDD.json` |
| [train_multi_factor_models.py](train_multi_factor_models.py) | weekly (Sunday) | `run_retrain_models.py` | Retrain 6 factor ensembles → overwrite `artifacts/ensemble_{factor}.pkl` |
| [backtest/run_backtest_multi_factor.py](backtest/run_backtest_multi_factor.py) | on demand | (manual) | Backtest using the same 6 pkl files |

### Occasional (W2-A workflow)

| Script | When to run | Purpose |
|---|---|---|
| [train_multi_horizon_models.py](train_multi_horizon_models.py) | When exploring horizon sweep | Train 18 ensembles (6 factors × 3 horizons) → `artifacts/multi_horizon/` |
| [promote_best_horizon_models.py](promote_best_horizon_models.py) | After the above | Select best-horizon per factor and overwrite the 6 production pkl. Auto-backs up prior pkl to `artifacts/pre_w2a_backup/` |
| [evaluate_w2a_horizons.py](evaluate_w2a_horizons.py) | Diagnostic | Compare win-rates across horizons |

### Supporting modules (imported, not entry points)

- [factors/factor_definitions.py](factors/factor_definitions.py), [factors/factor_weighting.py](factors/factor_weighting.py) — factor logic + IC-based weight resolver
- [features/](features/) — feature engineering (`build_dataset.py` is called by `prepare_prediction_data.py`)
- [data/universe.py](data/universe.py), [data/apply_liquidity_filter.py](data/apply_liquidity_filter.py) — universe management
- [config_loader.py](config_loader.py), [config_trend_filters.py](config_trend_filters.py) — reads `config.yaml`
- [update_selected_features.py](update_selected_features.py) — called by `prepare_prediction_data.py`

---

## Production artifacts (`artifacts/`)

Everything in `artifacts/` that is NOT under `deprecated/` is live.

### Models (6 factor ensembles)
```
ensemble_momentum.pkl
ensemble_trend.pkl
ensemble_volatility.pkl
ensemble_volume.pkl
ensemble_market.pkl
ensemble_alpha.pkl
```

Written by `train_multi_factor_models.py`. Loaded by `get_daily_signals_multi_factor.py` and `run_backtest_multi_factor.py`.

### Config
- `factor_weights.json` — per-factor IC-based weights (written by `train_multi_factor_models.py`, read by `factors/factor_weighting.py`)

### Datasets (train/test split for backtest)
- `train_dataset.parquet`, `test_dataset.parquet` — written by `features/build_dataset.py`

### Daily outputs (kept for audit)
- `signals_multi_factor_YYYYMMDD.json` — one per trading day

### Directories
- `multi_horizon/` — W2-A per-horizon training outputs
- `pre_w2a_backup/` — safety backup of prior production pkl (created by `promote_best_horizon_models.py`)
- `deprecated/` — archived experimental models (see below)

---

## What was archived (2026-07-18)

### `legacy/train/` (37 scripts)

Historical training experiments. None were reachable from the production entry points above. Kept for reference — recover via `git mv` if any becomes useful again.

Notable ones that may be reactivated:
- `optimize_hyperparams.py` — Optuna hyperparameter search
- `select_top_features.py` — feature-importance based feature selection
- `train_lstm.py`, `train_with_tabnet.py` — alternative model architectures
- `export_onnx.py` — ONNX export path (never wired into production)

### `legacy/` (9 old signal generators)

All predate the multi-factor pipeline. Loaded now-archived pkls (`ensemble_time_windows.pkl`, `xgboost_model.pkl`).

- `daily_trading_signals.py`, `get_daily_signals.py`, `generate_web_signals.py`
- `predict_all_stocks.py`, `predict_all_stocks_json.py`
- `get_stock_detail.py`, `list_features.py`
- `test_prediction_magnitude_accuracy.py`, `test_feb2_prediction.py`

### `artifacts/deprecated/` (~90 files)

Every model / metric / signal file that wasn't listed under "Production artifacts" above. Includes:
- All `xgboost_*.pkl`, `lightgbm_*.pkl`, `catboost_*.pkl` single-model artifacts
- All `ensemble_*_features*.pkl`, `ensemble_selected*`, `ensemble_ultimate*`, `ensemble_refined*`, `ensemble_time_windows*`, `super_ensemble*`, `stacking_*`, `optimized_60_percent_*`, `final_model_*` experiments
- Per-stock models (`stock_model_{GS,HD,META,NFLX,WMT}.pkl`)
- Pre-multi-factor signal files (`signals_20260131.json`, `signals_20260201.json`, `signals_weighted_20260201.json`)
- Orphan artifacts: `feature_manifest.json`, `refined_features_manifest.json`, `selected_features.json`, `thresholds.json`, `golden_vectors.json`, `model.onnx`, `backtest_results.png`

`.pkl` and `.json` files are gitignored, so this archiving is purely a local filesystem cleanup — nothing is lost from git history.

---

## Rules going forward

1. **New experiments go under `legacy/train/`** until validated. Only promote to `research/train_*.py` when wired into `run_retrain_models.py`.
2. **Every new production pkl gets a line here.** If it's not in this file, it's not production.
3. **Never load an artifact from `deprecated/` in production code.** If you find yourself needing to, promote it back out first and document why.
