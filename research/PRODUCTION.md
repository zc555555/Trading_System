# Production Assets — What's Live

This file is the single source of truth for "which scripts and models are actually in the daily/weekly production pipeline." Anything not listed here is either upstream tooling (data/features) or archived under `legacy/` / `artifacts/deprecated/`.

---

## Production entry points

| Script | Cadence | Called by | Purpose |
|---|---|---|---|
| [data/fetch_ohlcv.py](data/fetch_ohlcv.py) | daily / weekly | `run_auto_trading.py`, `run_retrain_models.py` | Pull latest OHLCV → `data/stocks.parquet` |
| [prepare_prediction_data.py](prepare_prediction_data.py) | daily | `run_auto_trading.py` | Rebuild features → `data/stocks_selected_features.parquet` |
| [get_daily_signals_multi_factor.py](get_daily_signals_multi_factor.py) | daily | `run_auto_trading.py`, `run_menu.py` | Load 6 factor ensembles → emit `artifacts/signals_multi_factor_YYYYMMDD.json` |
| [train_multi_factor_models.py](train_multi_factor_models.py) | weekly (Sunday) | `run_retrain_models.py` | Retrain 6 factor ensembles → overwrite `artifacts/ensemble_{factor}.pkl`. P1 (2026-07): uses the shared core in `evaluation/factor_training.py`; scores are per-date rank ICs on a purged inner validation; models refit on all data after weights are measured. |
| [backtest/run_backtest_multi_factor.py](backtest/run_backtest_multi_factor.py) | on demand | (manual) | Legacy quick backtest using the production pkls. ⚠️ Models are trained once on data overlapping the backtest window and no costs are applied — use the evaluation layer below for honest numbers. |

### Evaluation layer (P1, 2026-07) — the source of truth for performance claims

| Script | Purpose |
|---|---|
| [evaluation/purged_walk_forward.py](evaluation/purged_walk_forward.py) | Rolling 3y-train / 1q-test walk-forward with purge+embargo; retrains all 6 factor ensembles per fold; emits a fully out-of-sample prediction panel + per-date rank IC report (`evaluation/results/`) |
| [evaluation/simulate_portfolio.py](evaluation/simulate_portfolio.py) | Cost-aware simulation from that panel (next-open execution, 30 bps round trip): legacy 1-day and staggered 5-day modes, dev/holdout split at 2025-07-01 |
| [evaluation/factor_training.py](evaluation/factor_training.py) | Shared training core used by BOTH the walk-forward and `train_multi_factor_models.py` |
| [evaluation/metrics.py](evaluation/metrics.py) | Per-date cross-sectional rank IC + portfolio metrics |

Rules: performance numbers quoted anywhere (docs, decisions, README) must come from `evaluation/results/`, never from in-sample checks or the legacy backtest. The holdout segment (test windows ≥ 2025-07-01) must never drive tuning decisions.

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
- [data/build_edgar_fields.py](data/build_edgar_fields.py) — point-in-time share counts + filing calendar from the EDGAR cache (`edgar_fields.parquet`) for the mining DSL's auxiliary fields `marketcap` / `turnover` / `filing_days`. NOT in the nightly pipeline yet: if an adopted mined factor uses them, run `data/fetch_fundamentals.py` (refresh cache) + this script before `prepare_prediction_data.py` in `run_auto_trading.py` first. The same holds for the other mining fields: `data/gdelt_bigquery.py` (news), `data/build_form4_fields.py` after downloading the newest SEC insider-transactions quarter into `data/form345/`, and `data/fetch_finra_short_interest.py` (short interest); none of them refresh nightly. They refresh monthly through the `StockPredict_MiningSourcesRefresh` task (`scripts/run_refresh_sources_scheduled.bat`, 25th 19:00), and `StockPredict_SegmentReview` runs `scripts/run_review_scheduled.bat` on each rulebook review date.
- **Monthly (rulebook v3 ops):** rebuild the survivorship-complete panel (`experiments/survivorship_universe.py`, then `mining/harness.py baseline` per horizon) so the fresh evidence tier keeps growing; the panel is otherwise a manual artefact and `fresh` stalls at the last rebuild date.
- [factors/mined_factors.py](factors/mined_factors.py) + `factors/mined_factors.json` — registry of agent-mined factors adopted under rulebook track B; read by build_dataset, cross_sectional, update_selected_features and factor_definitions (empty registry = no-op). Written only by `mining/harness.py adopt`.
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
