"""
Create Refined Dataset: Optimized Feature Selection.

Strategy:
1. Remove useless features (importance = 0)
2. Keep only useful time windows (8 features)
3. Add top 5-10 advanced features (from 30)
4. Add 5 simple but critical new features

Target: 60 + 8 + 8 + 5 = 81 features
Expected: 59.24% -> 59.5-60%+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def add_simple_critical_features(df):
    """
    Add simple but potentially powerful features.

    Features:
    1. price_gap - Overnight gap
    2. volume_spike - Volume surge indicator
    3. intraday_range - Daily range
    4. close_position - Where close is relative to high/low
    5. consecutive_direction - Streak indicator
    """
    print("=" * 80)
    print("ADDING SIMPLE CRITICAL FEATURES")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    # 1. Price gap (overnight jump)
    print("\n1. Price Gap (overnight gap)")
    prev_close = df.groupby('symbol')['close'].shift(1)
    new_features['price_gap'] = (df['open'] - prev_close) / prev_close
    print(f"   Created: price_gap")

    # 2. Volume spike (current vs 20-day average)
    print("\n2. Volume Spike")
    vol_ma_20 = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    new_features['volume_spike'] = df['volume'] / vol_ma_20
    print(f"   Created: volume_spike")

    # 3. Intraday range
    print("\n3. Intraday Range")
    new_features['intraday_range'] = (df['high'] - df['low']) / df['close']
    print(f"   Created: intraday_range")

    # 4. Close position (0=at low, 1=at high)
    print("\n4. Close Position in Daily Range")
    daily_range = df['high'] - df['low']
    new_features['close_position'] = np.where(
        daily_range > 0,
        (df['close'] - df['low']) / daily_range,
        0.5
    )
    print(f"   Created: close_position")

    # 5. Consecutive direction strength
    print("\n5. Consecutive Direction Strength")
    returns = df.groupby('symbol')['close'].pct_change()

    def calc_streak(series):
        """Calculate current winning/losing streak."""
        if len(series) < 2:
            return 0
        current = series.iloc[-1]
        if pd.isna(current) or current == 0:
            return 0

        streak = 1
        direction = 1 if current > 0 else -1

        for i in range(len(series) - 2, -1, -1):
            val = series.iloc[i]
            if pd.isna(val):
                break
            if (val > 0 and direction > 0) or (val < 0 and direction < 0):
                streak += 1
            else:
                break

        return streak * direction

    new_features['consecutive_strength'] = returns.groupby(df['symbol']).transform(
        lambda x: x.rolling(10, min_periods=2).apply(calc_streak, raw=False)
    )
    print(f"   Created: consecutive_strength")

    print(f"\nCreated {len(new_features.columns)} simple critical features")

    return new_features


def main():
    """Create refined dataset with optimized features."""

    print("\n" + "=" * 80)
    print("REFINED DATASET CREATION")
    print("=" * 80)
    print("\nOptimizations:")
    print("  1. Remove 9 useless features (importance = 0)")
    print("  2. Keep 8 useful time window features")
    print("  3. Add top 8 advanced features (from 30)")
    print("  4. Add 5 simple critical features")
    print("\nTarget: ~81 features")
    print("Expected: 59.24% -> 59.5-60%+")

    data_dir = Path(__file__).parent.parent / "data"

    # Load base dataset (60 features)
    df_base = pd.read_parquet(data_dir / "stocks_selected_features.parquet")

    print(f"\n{'-' * 80}")
    print("STEP 1: LOAD BASE FEATURES")
    print(f"{'-' * 80}")
    print(f"\nBase dataset: {df_base.shape}")

    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    base_features = [c for c in df_base.columns if c not in meta_cols]
    print(f"Base features: {len(base_features)}")

    # Step 2: Add ONLY useful time window features
    print(f"\n{'-' * 80}")
    print("STEP 2: ADD USEFUL TIME WINDOW FEATURES")
    print(f"{'-' * 80}")

    useful_time_windows = [
        'roc_30d', 'volatility_15d', 'returns_30d', 'price_vs_ma_7d',
        'price_vs_ma_15d', 'volume_ratio_7d', 'roc_15d', 'returns_15d'
    ]

    print(f"\nUseful time window features (8):")
    for feat in useful_time_windows:
        print(f"  - {feat}")

    # Load time windows dataset and extract these features
    df_time_windows = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")
    df_useful_time_windows = df_time_windows[useful_time_windows]

    # Step 3: Add top advanced features
    print(f"\n{'-' * 80}")
    print("STEP 3: ADD TOP ADVANCED FEATURES")
    print(f"{'-' * 80}")

    # From ultimate model metrics, these advanced features were most important:
    top_advanced_features = [
        'interaction_oil_market_correlation',
        'interaction_dollar_oil_dynamics',
        'interaction_volatility_acceleration',
        'interaction_market_momentum_x_volatility',
        'vix_change_1d_y_squared',  # This is actually in base but listed
        'interaction_tech_industrial_spread',
        'interaction_detrended_market_sync',
        'interaction_short_long_momentum'
    ]

    print(f"\nTop advanced features to add (8):")

    df_enhanced = pd.read_parquet(data_dir / "stocks_enhanced_features.parquet")

    # Filter to only those that exist and aren't in base
    existing_advanced = []
    for feat in top_advanced_features:
        if feat in df_enhanced.columns and feat not in base_features:
            existing_advanced.append(feat)
            print(f"  - {feat}")

    df_top_advanced = df_enhanced[existing_advanced]

    # Step 4: Add simple critical features
    print(f"\n{'-' * 80}")
    print("STEP 4: ADD SIMPLE CRITICAL FEATURES")
    print(f"{'-' * 80}")

    df_simple = add_simple_critical_features(df_base)

    # Combine all
    print(f"\n{'-' * 80}")
    print("COMBINING ALL FEATURES")
    print(f"{'-' * 80}")

    df_refined = pd.concat([
        df_base,
        df_useful_time_windows,
        df_top_advanced,
        df_simple
    ], axis=1)

    # Remove NaN columns
    nan_threshold = 0.5
    nan_pct = df_refined.isna().mean()
    valid_cols_mask = nan_pct < nan_threshold

    # Get valid columns excluding meta
    all_feature_cols = [c for c in df_refined.columns if c not in meta_cols]
    valid_feature_cols = [c for c in all_feature_cols if valid_cols_mask[c]]

    print(f"\nFeature summary:")
    print(f"  Base features: {len(base_features)}")
    print(f"  Useful time windows: {len(useful_time_windows)}")
    print(f"  Top advanced features: {len(existing_advanced)}")
    print(f"  Simple critical features: {len(df_simple.columns)}")
    print(f"  Total before filtering: {len(all_feature_cols)}")
    print(f"  Total after NaN filtering: {len(valid_feature_cols)}")

    # Keep only valid features
    df_refined = df_refined[meta_cols + valid_feature_cols]

    print(f"\nRefined dataset shape: {df_refined.shape}")

    # Save
    output_path = data_dir / "stocks_refined_features.parquet"
    df_refined.to_parquet(output_path, index=False)

    print(f"\n{'-' * 80}")
    print(f"Refined dataset saved to: {output_path}")
    print(f"{'-' * 80}")

    # Save feature list
    feature_manifest = {
        'total_features': len(valid_feature_cols),
        'base_features': base_features,
        'useful_time_windows': useful_time_windows,
        'top_advanced_features': existing_advanced,
        'simple_critical_features': df_simple.columns.tolist(),
        'removed_useless_features': [
            'returns_3d', 'volatility_3d', 'volume_ratio_3d', 'roc_3d', 'price_vs_ma_3d',
            'returns_7d', 'volatility_7d', 'volume_ratio_15d', 'volatility_30d'
        ]
    }

    manifest_path = data_dir.parent / "artifacts" / "refined_features_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(feature_manifest, f, indent=2)

    print(f"Feature manifest saved to: {manifest_path}")

    print(f"\n{'=' * 80}")
    print("REFINED DATASET CREATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nImprovements:")
    print(f"  - Removed 9 useless features")
    print(f"  - Added 8 proven time window features")
    print(f"  - Added {len(existing_advanced)} top advanced features")
    print(f"  - Added 5 simple critical features")

    print(f"\nNext step: Train refined model")
    print(f"  python train/train_refined_model.py")
    print(f"\nExpected: 59.24% -> 59.5-60%+")


if __name__ == "__main__":
    main()
