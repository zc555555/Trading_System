"""
Complete test workflow to predict Feb 2 using Jan 30 data.

Steps:
1. Fetch latest data (should include Jan 30)
2. Process features while preserving the last day
3. Generate signals for Feb 2
"""

import sys
from pathlib import Path
import pandas as pd
import subprocess

def main():
    print("=" * 80)
    print("COMPLETE TEST: PREDICT FEB 2 USING JAN 30 DATA")
    print("=" * 80)

    # Step 0: Fetch latest raw data
    print("\n[Step 0/5] Fetching latest data from Yahoo Finance...")
    print("-" * 80)
    fetch_script = Path(__file__).parent / "data" / "fetch_ohlcv.py"
    result = subprocess.run([sys.executable, str(fetch_script)],
                          capture_output=True, text=True)

    # Show last few lines of output
    lines = result.stdout.split('\n')
    for line in lines[-10:]:
        if line.strip():
            print(line)

    # Verify we have Jan 30 data
    df_raw = pd.read_parquet('data/stocks.parquet')
    latest_date = df_raw['date'].max()
    print(f"\n[OK] Raw data fetched:")
    print(f"    Latest date: {latest_date.date()}")
    print(f"    Total rows: {len(df_raw):,}")

    if str(latest_date.date()) != '2026-01-30':
        print(f"\n[WARNING] Expected Jan 30, but got {latest_date.date()}")
        print("Continuing anyway with available data...")

    # IMPORTANT: Save raw data before processing overwrites it
    print("\n[Step 1/5] Saving raw data backup...")
    df_raw.to_parquet('data/stocks_raw_backup.parquet', index=False)
    print("[OK] Saved to data/stocks_raw_backup.parquet")

    # Step 1: Build features from raw data
    print("\n[Step 2/5] Building features (this will take a few minutes)...")
    print("-" * 80)

    sys.path.insert(0, str(Path(__file__).parent / "features"))
    from build_dataset import DatasetBuilder

    builder = DatasetBuilder()

    # Process raw data
    df = df_raw.copy()
    df_features = builder.compute_features(df)
    df_features = builder.compute_market_features(df_features)

    print(f"\n[OK] Features computed:")
    print(f"    Date range: {df_features['date'].min().date()} to {df_features['date'].max().date()}")

    # IMPORTANT: Save the last day BEFORE label generation drops it
    last_date = df_features['date'].max()
    last_day_features = df_features[df_features['date'] == last_date].copy()
    print(f"\n[INFO] Saved last day data ({last_date.date()}) with {len(last_day_features)} stocks")

    # Generate labels (this will DROP the last day)
    df_labeled = builder.generate_labels(df_features)
    print(f"After label generation: {df_labeled['date'].min().date()} to {df_labeled['date'].max().date()}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
    feature_cols = [c for c in df_labeled.columns if c not in meta_cols]

    # Clean dataset
    print("\n[Step 3/5] Cleaning dataset...")
    print("-" * 80)

    # Remove features with too many NaNs
    nan_pct = df_labeled[feature_cols].isna().mean()
    bad_features = nan_pct[nan_pct > 0.9].index.tolist()
    if bad_features:
        print(f"Dropping {len(bad_features)} features with >90% NaN")
        feature_cols = [f for f in feature_cols if f not in bad_features]

    # Fill NaNs per symbol
    df_labeled[feature_cols] = df_labeled.groupby('symbol')[feature_cols].ffill()
    df_labeled[feature_cols] = df_labeled.groupby('symbol')[feature_cols].bfill()
    df_labeled[feature_cols] = df_labeled[feature_cols].fillna(0)

    # NOW add back the last day for prediction
    print(f"\n[CRITICAL] Adding back last day ({last_date.date()}) for prediction...")

    # Set label and future_return to NaN for last day (we don't have future data)
    last_day_features['future_return'] = float('nan')
    last_day_features['label'] = float('nan')

    # Fill NaNs in features for last day
    last_day_features[feature_cols] = last_day_features.groupby('symbol')[feature_cols].ffill()
    last_day_features[feature_cols] = last_day_features.groupby('symbol')[feature_cols].bfill()
    last_day_features[feature_cols] = last_day_features[feature_cols].fillna(0)

    # Append last day to labeled data
    df_labeled = pd.concat([df_labeled, last_day_features], ignore_index=True)
    df_labeled = df_labeled.sort_values(['date', 'symbol']).reset_index(drop=True)

    print(f"\n[OK] Final dataset with last day restored:")
    print(f"    Date range: {df_labeled['date'].min().date()} to {df_labeled['date'].max().date()}")
    print(f"    Shape: {df_labeled.shape}")
    print(f"    Features: {len(feature_cols)}")

    # Verify last day
    last_day = df_labeled['date'].max()
    last_day_data = df_labeled[df_labeled['date'] == last_day]
    nan_labels = last_day_data['label'].isna().sum()
    print(f"    Last day ({last_day.date()}): {len(last_day_data)} stocks, {nan_labels} with NaN labels (EXPECTED for prediction)")

    # Save processed data with all dates
    df_labeled.to_parquet('data/stocks.parquet', index=False, compression='snappy')
    print(f"\n[OK] Saved processed data to data/stocks.parquet")

    # Step 2: Update selected features
    print("\n[Step 3/5] Updating selected features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "update_selected_features.py"
    subprocess.run([sys.executable, str(script_path)], check=True)

    # Step 3: Add time window features
    print("\n[Step 4/5] Adding time window features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "features" / "add_time_windows.py"
    result = subprocess.run([sys.executable, str(script_path)],
                          capture_output=True, text=True)
    # Show last few lines
    for line in result.stdout.split('\n')[-15:]:
        if line.strip():
            print(line)

    # Verify final data
    df_final = pd.read_parquet('data/stocks_with_time_windows.parquet')
    print(f"\n[OK] Final prediction data:")
    print(f"    Date range: {df_final['date'].min().date()} to {df_final['date'].max().date()}")
    print(f"    Shape: {df_final.shape}")

    # Step 4: Generate signals
    print("\n[Step 5/5] Generating trading signals for Feb 2...")
    print("=" * 80)
    script_path = Path(__file__).parent / "get_daily_signals.py"
    subprocess.run([sys.executable, str(script_path)])

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print(f"\nPrediction for Feb 2 (Monday) generated using {df_final['date'].max().date()} data")

if __name__ == "__main__":
    main()
