#!/usr/bin/env python
"""
Prepare latest data for prediction (including the most recent day).

This script:
1. Runs build_dataset to compute all features
2. PRESERVES the last day (even without labels) for prediction
3. Updates stocks_selected_features.parquet with the 60 base features
4. Adds time window features to create stocks_with_time_windows.parquet
5. (W1.6) Adds cross-sectional z-score features to the final parquet
"""

import sys
from pathlib import Path
import pandas as pd
import subprocess

# Add features to path
sys.path.insert(0, str(Path(__file__).parent / "features"))

from build_dataset import DatasetBuilder
from cross_sectional import add_cross_sectional_zscore, DEFAULT_XS_FEATURES

if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING PREDICTION DATA")
    print("=" * 80)

    # Step 1: Load raw data and build features
    print("\n[Step 1/3] Building features from raw data...")
    print("-" * 80)

    builder = DatasetBuilder()

    # Load raw data. Select ONLY the raw OHLCV columns so this script is
    # idempotent: re-running it on a file that already carries feature
    # columns (or _x/_y merge junk from older cycles) always starts from
    # the same clean input instead of enriching on top of enrichment.
    RAW_COLS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    df_raw = pd.read_parquet('data/stocks.parquet')
    missing = [c for c in RAW_COLS if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"stocks.parquet is missing raw columns {missing}")
    df_raw = df_raw[RAW_COLS]
    print(f"\n[OK] Loaded raw data:")
    print(f"    Date range: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
    print(f"    Shape: {df_raw.shape}")

    # Compute all features
    df_features = builder.compute_features(df_raw)
    df_features = builder.compute_market_features(df_features)

    print(f"\n[OK] Features computed:")
    print(f"    Date range: {df_features['date'].min().date()} to {df_features['date'].max().date()}")

    # CRITICAL: Save the last day BEFORE label generation drops it
    last_date = df_features['date'].max()
    last_day_features = df_features[df_features['date'] == last_date].copy()
    print(f"\n[INFO] Saved last day ({last_date.date()}) - {len(last_day_features)} stocks")

    # Generate labels (this will drop the last day because no future data exists)
    df_labeled = builder.generate_labels(df_features)
    print(f"[INFO] After labels: {df_labeled['date'].min().date()} to {df_labeled['date'].max().date()}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
    feature_cols = [c for c in df_labeled.columns if c not in meta_cols]

    # Clean dataset
    print("\n[INFO] Cleaning features...")
    nan_pct = df_labeled[feature_cols].isna().mean()
    bad_features = nan_pct[nan_pct > 0.9].index.tolist()
    if bad_features:
        print(f"  Dropping {len(bad_features)} features with >90% NaN")
        feature_cols = [f for f in feature_cols if f not in bad_features]

    # Fill NaNs: forward-fill only. Warm-up rows (before long-lookback
    # features have enough history) are DROPPED, not backward-filled --
    # bfill would copy future values into the past.
    df_labeled = df_labeled.sort_values(['symbol', 'date'])
    df_labeled[feature_cols] = df_labeled.groupby('symbol')[feature_cols].ffill()
    warmup = int(builder.config.get('features', {}).get('max_lookback', 250))
    obs_idx = df_labeled.groupby('symbol').cumcount()
    print(f"[INFO] Dropping {(obs_idx < warmup).sum():,} warm-up rows "
          f"(first {warmup} obs per symbol; replaces look-ahead bfill)")
    df_labeled = df_labeled[obs_idx >= warmup]
    df_labeled[feature_cols] = df_labeled[feature_cols].fillna(0)

    # CRITICAL: Add back the last day for prediction
    print(f"\n[CRITICAL] Restoring last day ({last_date.date()}) for prediction...")

    # Set labels to NaN (we don't have future data)
    last_day_features['future_return'] = float('nan')
    last_day_features['label'] = float('nan')

    # Fill NaNs in last day features (single-date frame: per-symbol
    # ffill/bfill are no-ops here, a plain 0-fill is all that applies)
    last_day_features[feature_cols] = last_day_features[feature_cols].fillna(0)

    # Append last day
    df_labeled = pd.concat([df_labeled, last_day_features], ignore_index=True)
    df_labeled = df_labeled.sort_values(['date', 'symbol']).reset_index(drop=True)

    print(f"\n[OK] Dataset ready:")
    print(f"    Date range: {df_labeled['date'].min().date()} to {df_labeled['date'].max().date()}")
    print(f"    Shape: {df_labeled.shape}")
    print(f"    Features: {len(feature_cols)}")

    # Save processed data to a SEPARATE file. stocks.parquet stays raw
    # (owned by fetch_ohlcv / apply_liquidity_filter); overwriting it with
    # enriched data made the pipeline non-idempotent and destroyed rows.
    df_labeled.to_parquet('data/stocks_features.parquet', index=False, compression='snappy')
    print(f"\n[OK] Saved to data/stocks_features.parquet")

    # Step 2: Update stocks_selected_features.parquet
    print("\n[Step 2/3] Updating selected features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "update_selected_features.py"
    subprocess.run([sys.executable, str(script_path)], check=True)

    # Step 3: Add time window features
    print("\n[Step 3/3] Adding time window features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "features" / "add_time_windows.py"
    result = subprocess.run([sys.executable, str(script_path)],
                          capture_output=True, text=True)
    # Show summary
    for line in result.stdout.split('\n')[-10:]:
        if line.strip():
            print(line)

    # Step 4: Add cross-sectional features (W1.6, 2026-05)
    print("\n[Step 4/4] Adding cross-sectional z-score features...")
    print("-" * 80)
    df_pre_xs = pd.read_parquet('data/stocks_with_time_windows.parquet')
    print(f"Pre-xs shape: {df_pre_xs.shape}")

    df_xs = add_cross_sectional_zscore(
        df_pre_xs,
        features=DEFAULT_XS_FEATURES,
        suffix='_xs',
        winsorize_pct=0.01,
    )
    df_xs.to_parquet('data/stocks_with_time_windows.parquet', index=False, compression='snappy')
    print(f"Post-xs shape: {df_xs.shape}")

    # Verify final result
    df_final = df_xs
    print("\n" + "=" * 80)
    print("PREDICTION DATA READY")
    print("=" * 80)
    print(f"\n[OK] stocks_with_time_windows.parquet:")
    print(f"    Date range: {df_final['date'].min().date()} to {df_final['date'].max().date()}")
    print(f"    Shape: {df_final.shape}")
    total_features = len([c for c in df_final.columns
                         if c not in ['date', 'symbol', 'open', 'high', 'low',
                                    'close', 'volume', 'future_return', 'label']])
    xs_features = len([c for c in df_final.columns if c.endswith('_xs')])
    print(f"    Total features: {total_features}  (incl. {xs_features} cross-sectional _xs columns)")
    print(f"\nReady to generate signals using {df_final['date'].max().date()} data!")
