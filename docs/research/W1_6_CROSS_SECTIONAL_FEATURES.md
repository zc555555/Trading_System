# W1.6 — Cross-Sectional Feature Engineering for TREND / VOLUME / VOLATILITY

**Status**: shipped 2026-05-18. Companion to W1 and W1.5.

W1.5 revealed that the TREND and VOLUME factor models had **negative** validation IC on the 302-stock universe (-0.012 and -0.016 respectively). Under IC-proportional weighting they got zero weight, so they no longer harmed signals — but they still consumed training compute and disk space. W1.6 attempts to actually *fix* them by rebuilding their features as cross-sectional z-scores.

---

## TL;DR honest result

| factor | W1.5 IC | **W1.6 IC** | what changed |
|---|---|---|---|
| momentum | +0.2609 | +0.2609 | unchanged (intentionally — see below) |
| alpha | +0.0667 | +0.0667 | unchanged |
| volatility | +0.0237 | **+0.0255** ↑ | +5 xs features added |
| volume | **-0.0161** | **+0.0004** ↑ | all-features rewritten to xs |
| trend | **-0.0117** | **-0.0027** ↑ | all-features rewritten to xs |
| market | -0.0060 | -0.0060 | unchanged (structural, fix in W2) |
| **overall win rate (last 500)** | **63.4%** | **63.4%** | **unchanged** |

W1.6 **fixed the anti-predictive factors** (trend, volume) but **did not improve the signal**. The reason is that momentum at 73.8% weight dominates the blend, and momentum was already optimal. The other factors getting cleaner just made the architecture tidier and slightly more diversified.

This is a structural win, not an accuracy win. The next gain has to come from changing the prediction *horizon* (currently 1-day), not the features.

---

## The diagnosis (why the old features failed)

A factor that wants to predict cross-sectional returns must produce features that **vary across stocks on the same date**. Three failures in the old definitions:

### TREND features

Old TREND included:
- `golden_cross`, `price_above_sma_20`, `price_above_ema_50`, `adx_strong_trend` — all **binary** flags that split the universe ~50/50. No ranking.
- `price_vs_ma_3d`, `price_vs_ma_30d` etc. — continuous but **absolute** (e.g., "+5% above MA30"). The model doesn't know if +5% is good or bad relative to a peer at +12%.

### VOLUME features

Old VOLUME had:
- `volume_increasing` — binary noise.
- `volume_ratio_20d` — per-stock against its own history. Doesn't tell you if this stock is having an unusually big volume burst *relative to peers today*.

### MARKET features (still broken, deferred)

All market features (`spy_returns_1d_y`, `vix_level_y`, etc.) are **identical across all 302 stocks on the same date** (std = 0, unique = 1 per date). A cross-sectional model literally cannot extract signal from identical features. Fixing this requires constructing interaction features (`per_stock_feature × market_state`) — that's W2 work.

---

## The fix

### New module — `research/features/cross_sectional.py`

Two utilities for converting per-stock features to per-date rankings:

```python
add_cross_sectional_zscore(df, features, suffix='_xs', winsorize_pct=0.01)
    # For each (date, feature): clip to [1%, 99%] quantile, then z-score within date
    # Result: feature_xs has mean ~0, std ~1 within each trading day

add_cross_sectional_rank(df, features, suffix='_xrank')
    # For each (date, feature): percentile rank in [0, 1]
    # Robust to outliers, throws away magnitude
```

A `DEFAULT_XS_FEATURES` list ships 42 features that benefit from xs treatment (returns at all horizons, ROCs, volume ratios, volatility windows, oscillators, MAs).

### Wired into the pipeline

`prepare_prediction_data.py` now has a **Step 4** that loads the time-windows parquet, calls `add_cross_sectional_zscore` on all 42 default features, and saves back. Adds ~30% column count (89 → 131) but the parquet file size barely grows (z-scores compress well).

### Factor redefinitions

**`research/factors/factor_definitions.py`** changes:

```diff
TREND_FEATURES = [
-   'adx_strong_trend', 'supertrend',          # binary noise
-   'golden_cross', 'price_above_sma_20',      # binary noise
-   'price_above_ema_50',                      # binary noise
-   'price_vs_ma_3d', 'price_vs_ma_7d',        # absolute features
-   'price_vs_ma_15d', 'price_vs_ma_30d',
-   'psar',
+   'price_vs_ma_3d_xs',
+   'price_vs_ma_7d_xs',
+   'price_vs_ma_15d_xs',
+   'price_vs_ma_30d_xs',
+   'macd_xs',
+   'macd_hist_xs',
]

VOLUME_FEATURES = [
-   'volume_ratio_3d', 'volume_ratio_5d',      # absolute
-   'volume_ratio_7d', 'volume_ratio_15d',
-   'volume_ratio_20d', 'volume_ratio_30d',
-   'volume_std_20d',
-   'volume_increasing',                       # binary noise
-   'vwap_ratio',
+   'volume_ratio_3d_xs',
+   'volume_ratio_5d_xs',
+   'volume_ratio_7d_xs',
+   'volume_ratio_15d_xs',
+   'volume_ratio_20d_xs',
+   'volume_ratio_30d_xs',
+   'volume_std_20d_xs',
+   'vwap_ratio_xs',
]

VOLATILITY_FEATURES = [
    'volatility_3d', 'volatility_10d',         # kept (already worked)
    'volatility_20d', 'volatility_60d',
    'bb_width', 'bb_squeeze', 'kc_position',
+   # New xs additions
+   'volatility_3d_xs',
+   'volatility_10d_xs',
+   'volatility_20d_xs',
+   'volatility_60d_xs',
+   'bb_width_xs',
]
```

`MOMENTUM_FEATURES`, `ALPHA_FEATURES`, `MARKET_FEATURES` unchanged. (See "Negative result" below for momentum.)

---

## Negative result: adding xs features to MOMENTUM **hurt**

Hypothesis tested: append 20 xs features (`returns_5d_xs`, `roc_15d_xs`, `rsi_7_xs`, ...) to MOMENTUM to capture cross-sectional momentum (Jegadeesh-Titman style).

Result: IC dropped from **0.2609 -> 0.2577** (-0.003).

Interpretation: at 27 features and 600k samples, momentum was already saturated for the GBM-ensemble architecture. Adding 20 more features added noise faster than signal, even though the new features were "textbook correct". This is a textbook feature-bloat regression.

Lesson: do not paper over a saturated model with more features. To push momentum past 0.26 IC needs either (a) a different model class (e.g., LambdaRank), or (b) richer features that aren't redundant with the existing 27 (e.g., earnings drift, IV skew). Both are W2+ work.

**Action taken**: reverted MOMENTUM to its W1 definition. Kept the new `_xs` columns in the parquet for future use by other consumers.

---

## What this DID NOT solve

- **Signal accuracy unchanged** — overall win rate stayed at 63.4%, SELL at 69.2%, BUY at 57.0%. The blend's effective IC didn't move because momentum dominates and momentum didn't change.
- **MARKET factor still broken** — its features have zero cross-sectional variance by construction. Need interaction features in W2.
- **TREND still slightly negative (-0.003)** — xs rescued it from -0.012 but it's still not productive at the 1-day prediction horizon. Trend signals are inherently multi-day.
- **VOLUME barely positive (+0.0004)** — same story. Volume bursts predict 5d+ moves, not 1d.

---

## What this enables for W2

- The `_xs` columns now sit in the parquet ready to be used by new factor groups (e.g. a `reversal` factor using `returns_2d_xs` with a negative-sign expectation).
- The `cross_sectional` module is reusable for any future feature transformation — IV factors, GARCH features, etc.
- The "saturate then add" recipe is validated: don't bulk-load features into already-working factors; create *new* factor groups with the new features and let IC-proportional weighting decide.

---

## Run the new pipeline

```bash
cd research
./venv/Scripts/python.exe data/fetch_ohlcv.py           # if data is stale
./venv/Scripts/python.exe data/apply_liquidity_filter.py
./venv/Scripts/python.exe prepare_prediction_data.py    # now includes Step 4 (xs)
./venv/Scripts/python.exe train_multi_factor_models.py  # picks up new feature lists
./venv/Scripts/python.exe get_daily_signals_multi_factor.py
```

If you already have a recent parquet and only want to add xs columns:

```bash
./venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'features')
import pandas as pd
from cross_sectional import add_cross_sectional_zscore, DEFAULT_XS_FEATURES
df = pd.read_parquet('data/stocks_with_time_windows.parquet')
df = add_cross_sectional_zscore(df, DEFAULT_XS_FEATURES, suffix='_xs', winsorize_pct=0.01)
df.to_parquet('data/stocks_with_time_windows.parquet', index=False)
"
```

---

## Rollback

If for some reason the xs-based TREND/VOLUME hurt production performance:

1. Edit `factors/factor_definitions.py` and restore the pre-W1.6 lists from git history (or this doc's diff block).
2. Re-run `train_multi_factor_models.py`.
3. The `_xs` columns in the parquet are harmless if not referenced.

Or skip xs entirely: set `factor_weighting.strategy: "static"` in `config.yaml` to revert to hand-tuned weights and the old factor lists become irrelevant.

---

## What to do next

Based on this experiment, two paths forward:

**Path A — change the prediction horizon (recommended)**.
The 1-day horizon is the actual bottleneck. Many of the W1.6 features that "failed" probably work at 5- or 20-day horizons. Add new factor models that predict `future_return_5d` and `future_return_20d`, then blend across horizons. Expected gain: large.

**Path B — add new feature sources (W2 plan)**.
GARCH volatility forecasts, IV skew, earnings surprise, analyst revisions. Net-new information, not transformations of existing data. Expected gain: large.

Path B is the original W2 in the master plan. Path A is a smaller diversion that could move the needle on the existing features. The user (you) should pick.
