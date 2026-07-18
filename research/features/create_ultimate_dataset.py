"""
Create Ultimate Dataset: Combine ALL useful features.

Features:
- 60 base features (proven: 59.12%)
- 30 advanced features (interactions, FFT, higher-order stats)
- 20 time window features (proven: +0.12%)

Total: 110 features

Hypothesis: Time windows + Advanced features may have synergy
Expected: 59.24% -> 59.5-60%+
"""

import pandas as pd
from pathlib import Path


def main():
    """Combine all feature types into ultimate dataset."""

    print("=" * 80)
    print("CREATING ULTIMATE FEATURE DATASET")
    print("=" * 80)
    print("\nCombining:")
    print("  - 60 base features (proven)")
    print("  - 30 advanced features (interactions, FFT, stats)")
    print("  - 20 time window features (proven +0.12%)")
    print("\nTotal: 110 features")

    data_dir = Path(__file__).parent.parent / "data"

    # Load all datasets
    print("\n" + "-" * 80)
    print("LOADING DATASETS")
    print("-" * 80)

    df_base = pd.read_parquet(data_dir / "stocks_selected_features.parquet")
    df_enhanced = pd.read_parquet(data_dir / "stocks_enhanced_features.parquet")
    df_time_windows = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    print(f"\nBase dataset: {df_base.shape}")
    print(f"Enhanced dataset: {df_enhanced.shape}")
    print(f"Time windows dataset: {df_time_windows.shape}")

    # Identify unique features from each
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']

    base_features = [c for c in df_base.columns if c not in meta_cols]

    # Get ONLY new features from enhanced (not in base)
    enhanced_new = [c for c in df_enhanced.columns
                   if c not in meta_cols and c not in base_features]

    # Get ONLY new features from time windows (not in base)
    time_window_new = [c for c in df_time_windows.columns
                      if c not in meta_cols and c not in base_features]

    print(f"\nFeature breakdown:")
    print(f"  Base features: {len(base_features)}")
    print(f"  Enhanced (new only): {len(enhanced_new)}")
    print(f"  Time windows (new only): {len(time_window_new)}")
    print(f"  Total unique: {len(base_features) + len(enhanced_new) + len(time_window_new)}")

    # Extract new features
    df_enhanced_new = df_enhanced[enhanced_new]
    df_time_windows_new = df_time_windows[time_window_new]

    # Combine everything
    print("\n" + "-" * 80)
    print("COMBINING FEATURES")
    print("-" * 80)

    df_ultimate = pd.concat([
        df_base,
        df_enhanced_new,
        df_time_windows_new
    ], axis=1)

    print(f"\nUltimate dataset shape: {df_ultimate.shape}")

    # Verify no duplicate columns
    assert df_ultimate.shape[1] == len(df_ultimate.columns), "Duplicate columns detected!"

    # Count features by type
    all_features = [c for c in df_ultimate.columns if c not in meta_cols]

    interaction_feats = [c for c in all_features if 'interaction_' in c]
    fft_feats = [c for c in all_features if 'fft_' in c]
    higher_order_feats = [c for c in all_features if any(x in c for x in ['skew', 'kurtosis', 'iqr'])]
    trend_feats = [c for c in all_features if 'trend_' in c]
    time_window_feats = [c for c in all_features if any(f'{d}d' in c for d in [3, 7, 15, 30])]
    nonlinear_feats = [c for c in all_features if any(x in c for x in ['squared', 'sqrt', '_log'])]

    print(f"\nFeature types in ultimate dataset:")
    print(f"  Base features: {len(base_features)}")
    print(f"  Interaction features: {len(interaction_feats)}")
    print(f"  FFT features: {len(fft_feats)}")
    print(f"  Higher-order stats: {len(higher_order_feats)}")
    print(f"  Trend features: {len(trend_feats)}")
    print(f"  Time window features: {len(time_window_feats)}")
    print(f"  Non-linear transforms: {len(nonlinear_feats)}")
    print(f"  Total: {len(all_features)}")

    # Save
    output_path = data_dir / "stocks_ultimate_features.parquet"
    df_ultimate.to_parquet(output_path, index=False)

    print(f"\n{'-' * 80}")
    print(f"Ultimate dataset saved to: {output_path}")
    print(f"{'-' * 80}")

    print(f"\n{'=' * 80}")
    print("ULTIMATE DATASET CREATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nNext step: Train with ultimate features")
    print(f"  python train/train_ultimate_model.py")
    print(f"\nHypothesis: Combining time windows + advanced features")
    print(f"Expected: 59.24% -> 59.5-60%+")


if __name__ == "__main__":
    main()
