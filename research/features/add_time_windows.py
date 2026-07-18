"""
Add Missing Time Window Features (3d, 7d, 15d, 30d).

Current: 1d, 5d, 10d, 20d, 50d, 60d
Missing: 3d, 7d, 15d, 30d

Key hypothesis: 7-day (1 week) and 30-day (1 month) windows are
important trading cycles that may provide unique information.

Expected: 59.12% -> 59.5-60%+ (+0.4-0.9%)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_time_window_features(df, windows=[3, 7, 15, 30]):
    """
    Add returns and volatility features for specified time windows.

    For each window:
    - returns_{window}d: Price return over window
    - volatility_{window}d: Return volatility over window
    """
    print("=" * 80)
    print("ADDING TIME WINDOW FEATURES")
    print("=" * 80)

    print(f"\nTime windows to add: {windows}")

    new_features = pd.DataFrame(index=df.index)

    for window in windows:
        print(f"\nAdding {window}-day features:")

        # 1. Returns over window
        returns_col = f'returns_{window}d'
        new_features[returns_col] = df.groupby('symbol')['close'].pct_change(window)
        print(f"  Created: {returns_col}")

        # 2. Volatility over window
        volatility_col = f'volatility_{window}d'
        # Calculate rolling std of 1-day returns over the window
        daily_returns = df.groupby('symbol')['close'].pct_change()
        new_features[volatility_col] = daily_returns.groupby(df['symbol']).transform(
            lambda x: x.rolling(window, min_periods=max(2, window//2)).std()
        )
        print(f"  Created: {volatility_col}")

        # 3. Volume ratio over window
        volume_col = f'volume_ratio_{window}d'
        new_features[volume_col] = df.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(window, min_periods=max(2, window//2)).mean()
        )
        print(f"  Created: {volume_col}")

    print(f"\nTotal new features: {len(new_features.columns)}")

    return new_features


def add_market_time_windows(df, windows=[3, 7, 15, 30]):
    """
    Add market index features for specified time windows.

    For each window, add returns for:
    - SPY, QQQ, DIA, IWM (market indices)
    - VIX change
    - TLT, GLD, USO (cross-asset)
    """
    print("\n" + "=" * 80)
    print("ADDING MARKET TIME WINDOW FEATURES")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    # Market columns that exist
    market_bases = [
        'spy_returns', 'qqq_returns', 'dia_returns', 'iwm_returns',
        'vix_change', 'tlt_returns', 'gld_returns', 'uso_returns', 'uup_returns'
    ]

    for window in windows:
        print(f"\nAdding {window}-day market features:")

        for base in market_bases:
            # Try to find the base feature in different suffixes
            # Look for _y, _x, or no suffix
            source_col = None
            for suffix in ['_y', '_x', '']:
                for period in ['_1d', '_5d', '']:  # Try different existing periods
                    potential_col = f'{base}{period}{suffix}'
                    if potential_col in df.columns:
                        source_col = potential_col
                        break
                if source_col:
                    break

            if not source_col:
                continue

            # Create new window feature name
            new_col = f'{base}_{window}d_y'

            # Calculate feature based on window
            # We'll use the 1d version if available and calculate the window
            if '_1d' in source_col or 'returns_1d' in source_col:
                # It's already a 1-day return, we can compute multi-day by rolling product
                # But simpler: just create as-is for now
                # Skip if we can't properly calculate
                continue
            else:
                # Copy with new name (as proxy)
                new_features[new_col] = df[source_col]
                print(f"  Created: {new_col} (from {source_col})")

    print(f"\nTotal new market features: {len(new_features.columns)}")

    return new_features


def add_momentum_features(df, windows=[3, 7, 15, 30]):
    """
    Add momentum features for time windows.

    - ROC (Rate of Change)
    - Momentum oscillator
    """
    print("\n" + "=" * 80)
    print("ADDING MOMENTUM FEATURES")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    for window in windows:
        print(f"\nAdding {window}-day momentum features:")

        # Rate of Change (ROC)
        roc_col = f'roc_{window}d'
        new_features[roc_col] = df.groupby('symbol')['close'].transform(
            lambda x: (x - x.shift(window)) / x.shift(window) * 100
        )
        print(f"  Created: {roc_col}")

        # Price vs Moving Average
        price_vs_ma_col = f'price_vs_ma_{window}d'
        ma = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window, min_periods=max(2, window//2)).mean()
        )
        new_features[price_vs_ma_col] = (df['close'] - ma) / ma
        print(f"  Created: {price_vs_ma_col}")

    print(f"\nTotal new momentum features: {len(new_features.columns)}")

    return new_features


def main():
    """Add time window features to dataset."""

    print("\n" + "=" * 80)
    print("TIME WINDOW FEATURE GENERATION")
    print("=" * 80)
    print("\nMissing time windows: 3d, 7d, 15d, 30d")
    print("Target: Add ~20-30 new time window features")
    print("Expected: 59.12% -> 59.5-60%+")

    # Load dataset
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_selected_features.parquet")

    print(f"\n{'-' * 80}")
    print("LOADING BASE DATASET")
    print(f"{'-' * 80}")
    print(f"\nOriginal shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    existing_features = [c for c in df.columns if c not in meta_cols]
    print(f"Existing features: {len(existing_features)}")

    # Add features
    all_new_features = []

    # 1. Basic time window features
    time_window_feats = add_time_window_features(df, windows=[3, 7, 15, 30])
    all_new_features.append(time_window_feats)

    # 2. Momentum features
    momentum_feats = add_momentum_features(df, windows=[3, 7, 15, 30])
    all_new_features.append(momentum_feats)

    # Combine all new features
    print("\n" + "=" * 80)
    print("COMBINING FEATURES")
    print("=" * 80)

    df_new_features = pd.concat(all_new_features, axis=1)

    # Remove columns with too many NaNs
    nan_threshold = 0.5
    nan_pct = df_new_features.isna().mean()
    valid_cols = nan_pct[nan_pct < nan_threshold].index.tolist()

    print(f"\nNew features created: {len(df_new_features.columns)}")
    print(f"Valid features (< 50% NaN): {len(valid_cols)}")
    print(f"Removed features (too many NaN): {len(df_new_features.columns) - len(valid_cols)}")

    df_new_features = df_new_features[valid_cols]

    # Combine with original
    df_with_time_windows = pd.concat([df, df_new_features], axis=1)

    print(f"\nDataset with time windows:")
    print(f"  Original features: {len(existing_features)}")
    print(f"  New features: {len(valid_cols)}")
    print(f"  Total features: {len(existing_features) + len(valid_cols)}")
    print(f"  Total shape: {df_with_time_windows.shape}")

    # Save
    output_path = data_dir / "stocks_with_time_windows.parquet"
    df_with_time_windows.to_parquet(output_path, index=False)

    print(f"\n{'-' * 80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'-' * 80}")

    # Show breakdown
    print(f"\nNew features by window:")
    for window in [3, 7, 15, 30]:
        count = len([c for c in valid_cols if f'{window}d' in c])
        print(f"  {window:2d}-day: {count} features")

    print(f"\n{'=' * 80}")
    print("TIME WINDOW FEATURE GENERATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nNext step: Train with time window features")
    print(f"  python train/train_with_time_windows.py")
    print(f"\nExpected improvement: +0.4-0.9%")
    print("Key hypothesis: 7-day (1 week) cycle is critical for stock prediction")


if __name__ == "__main__":
    main()
