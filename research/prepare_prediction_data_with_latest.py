"""
Prepare prediction data INCLUDING the very latest day (even without labels).

For prediction, we need the most recent data to make forecasts.
This script ensures Jan 30 data is included for predicting Feb 2.
"""

import sys
from pathlib import Path
import pandas as pd
import subprocess

# Add features to path
sys.path.insert(0, str(Path(__file__).parent / "features"))

from build_dataset import DatasetBuilder

if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING PREDICTION DATA (WITH LATEST DAY)")
    print("=" * 80)

    # Step 1: Build dataset - this will drop the last day due to label generation
    print("\n[Step 1/4] Building dataset with features...")
    print("-" * 80)
    builder = DatasetBuilder()
    builder.config['data']['parquet_path'] = 'data/stocks.parquet'

    # Load and compute features (without dropping last day yet)
    df = builder.load_data()
    print(f"\nRaw data loaded: {df['date'].min()} to {df['date'].max()}")

    df_features = builder.compute_features(df)
    df_features = builder.compute_market_features(df_features)
    print(f"After features: {df_features['date'].min()} to {df_features['date'].max()}")

    # Generate labels (this will create NaN for last day)
    df_labeled = builder.generate_labels(df_features)
    print(f"After labels: {df_labeled['date'].min()} to {df_labeled['date'].max()}")

    # Get feature columns
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
    feature_cols = [c for c in df_labeled.columns if c not in meta_cols]

    # Clean dataset but DON'T drop last day
    print("\nCleaning dataset (preserving last day for prediction)...")
    df_clean, feature_cols_clean = builder.clean_dataset(df_labeled, feature_cols)

    # IMPORTANT: Add back the last day even if it has NaN label
    last_day_mask = df_labeled['date'] == df_labeled['date'].max()
    if last_day_mask.sum() > 0 and df_labeled.loc[last_day_mask, 'label'].isna().all():
        print(f"\n[INFO] Last day {df_labeled['date'].max().date()} has no labels (expected)")
        print("       Adding it back for prediction purposes...")

        # Get last day data
        last_day_data = df_labeled[last_day_mask].copy()

        # Fill NaN features (same as clean_dataset does)
        last_day_data[feature_cols_clean] = last_day_data.groupby('symbol')[feature_cols_clean].ffill()
        last_day_data[feature_cols_clean] = last_day_data.groupby('symbol')[feature_cols_clean].bfill()
        last_day_data[feature_cols_clean] = last_day_data[feature_cols_clean].fillna(0)

        # Append to cleaned data
        df_clean = pd.concat([df_clean, last_day_data], ignore_index=True)
        df_clean = df_clean.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Save to stocks.parquet
    parquet_path = Path(__file__).parent / 'data' / 'stocks.parquet'
    df_clean.to_parquet(parquet_path, index=False, compression='snappy')
    print(f"\n[OK] Saved to: {parquet_path}")
    print(f"    Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"    Shape: {df_clean.shape}")

    # Step 2: Update stocks_selected_features.parquet
    print("\n[Step 2/4] Updating selected features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "update_selected_features.py"
    subprocess.run([sys.executable, str(script_path)], check=True)

    # Step 3: Add time window features
    print("\n[Step 3/4] Adding time window features...")
    print("-" * 80)
    script_path = Path(__file__).parent / "features" / "add_time_windows.py"
    subprocess.run([sys.executable, str(script_path)], check=True)

    # Step 4: Verify final result
    print("\n[Step 4/4] Verifying final dataset...")
    print("-" * 80)
    df_final = pd.read_parquet('data/stocks_with_time_windows.parquet')
    print("\n" + "=" * 80)
    print("PREDICTION DATA READY")
    print("=" * 80)
    print(f"\n[OK] stocks_with_time_windows.parquet:")
    print(f"    Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"    Shape: {df_final.shape}")
    print(f"    Latest date stocks: {len(df_final[df_final['date'] == df_final['date'].max()])}")

    total_features = len([c for c in df_final.columns
                         if c not in ['date', 'symbol', 'open', 'high', 'low',
                                    'close', 'volume', 'future_return', 'label']])
    print(f"    Total features: {total_features}")

    print(f"\nReady to predict Feb 2 using {df_final['date'].max().date()} data!")
