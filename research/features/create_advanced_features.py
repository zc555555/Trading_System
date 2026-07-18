"""
Create Advanced Features to Break 60% Barrier.

New feature types (not in original 175):
1. Feature Interactions: Top features crossed (multiplication)
2. Higher-order Statistics: Skewness, Kurtosis
3. Time Series Decomposition: Trend, Seasonality components
4. Frequency Domain: FFT components
5. Non-linear Transformations: Log-returns, Sqrt, Power features

Strategy: Add 20-30 high-quality NEW features to the existing 60
Expected: 59.12% -> 59.5-60.2% (+0.4-1.0%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler


def create_feature_interactions(df, top_features, max_interactions=15):
    """
    Create feature interaction terms (multiplication of top features).

    Args:
        df: DataFrame with existing features
        top_features: List of most important features
        max_interactions: Maximum number of interaction terms to create

    Returns:
        DataFrame with new interaction features
    """
    print("\n" + "=" * 80)
    print("CREATING FEATURE INTERACTIONS")
    print("=" * 80)

    print(f"\nTop features for interaction:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i:2d}. {feat}")

    new_features = pd.DataFrame(index=df.index)

    # Select most promising interactions
    # Strategy: Combine different types of features
    interactions = [
        # Market momentum x Volatility
        ('spy_returns_1d_y', 'vix_change_1d_y', 'market_momentum_x_volatility'),
        ('qqq_returns_1d_y', 'vix_level_y', 'tech_momentum_x_fear'),

        # Commodities x Market
        ('uso_returns_1d_y', 'spy_returns_1d_y', 'oil_market_correlation'),
        ('gld_returns_1d_y', 'vix_level_y', 'gold_fear_correlation'),
        ('tlt_returns_1d_y', 'spy_returns_1d_y', 'bonds_stocks_divergence'),

        # Technical x Market
        ('dpo_20', 'spy_returns_5d_y', 'detrended_market_sync'),
        ('mfi_14', 'volume_ratio_20d', 'money_flow_volume'),

        # Cross-asset strength
        ('dia_returns_1d_y', 'iwm_returns_1d_y', 'large_small_cap_spread'),
        ('qqq_returns_1d_y', 'dia_returns_1d_y', 'tech_industrial_spread'),

        # Currency x Commodities
        ('uup_returns_1d_y', 'gld_returns_1d_y', 'dollar_gold_inverse'),
        ('uup_returns_1d_y', 'uso_returns_1d_y', 'dollar_oil_dynamics'),

        # Multi-timeframe
        ('spy_returns_1d_y', 'spy_returns_20d_y', 'short_long_momentum'),
        ('vix_change_1d_y', 'vix_change_5d_y', 'volatility_acceleration'),

        # Breadth x Returns
        ('market_breadth_1d', 'spy_returns_1d_y', 'breadth_return_divergence'),
    ]

    created_count = 0
    for feat1, feat2, name in interactions[:max_interactions]:
        if feat1 in df.columns and feat2 in df.columns:
            new_features[f'interaction_{name}'] = df[feat1] * df[feat2]
            created_count += 1
            print(f"  Created: {name:40s} ({feat1} × {feat2})")

    print(f"\nCreated {created_count} interaction features")

    return new_features


def create_higher_order_stats(df, price_col='close', volume_col='volume', windows=[5, 10, 20]):
    """
    Create higher-order statistical features.

    Features:
    - Skewness: Measure of asymmetry
    - Kurtosis: Measure of tail heaviness
    - Quantile ranges
    """
    print("\n" + "=" * 80)
    print("CREATING HIGHER-ORDER STATISTICS")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    # Calculate returns if not present
    if 'returns_1d' not in df.columns:
        returns = df.groupby('symbol')[price_col].pct_change()
    else:
        returns = df['returns_1d']

    for window in windows:
        print(f"\nWindow: {window} days")

        # Skewness of returns
        skew_col = f'returns_skew_{window}d'
        new_features[skew_col] = returns.groupby(df['symbol']).transform(
            lambda x: x.rolling(window, min_periods=max(3, window//2)).skew()
        )
        print(f"  Created: {skew_col}")

        # Kurtosis of returns
        kurt_col = f'returns_kurtosis_{window}d'
        new_features[kurt_col] = returns.groupby(df['symbol']).transform(
            lambda x: x.rolling(window, min_periods=max(3, window//2)).apply(stats.kurtosis, raw=True)
        )
        print(f"  Created: {kurt_col}")

        # Quantile range (robust measure of dispersion)
        q75_q25_col = f'returns_iqr_{window}d'
        new_features[q75_q25_col] = returns.groupby(df['symbol']).transform(
            lambda x: x.rolling(window, min_periods=max(3, window//2)).apply(
                lambda y: np.percentile(y, 75) - np.percentile(y, 25) if len(y) >= 3 else np.nan
            )
        )
        print(f"  Created: {q75_q25_col}")

    print(f"\nCreated {len(new_features.columns)} higher-order stat features")

    return new_features


def create_trend_features(df, price_col='close', windows=[10, 20, 50]):
    """
    Create trend decomposition features.

    Features:
    - Linear trend strength
    - Trend acceleration (2nd derivative)
    - Deviation from trend
    """
    print("\n" + "=" * 80)
    print("CREATING TREND FEATURES")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    for window in windows:
        print(f"\nWindow: {window} days")

        # Linear trend strength (R-squared of linear fit)
        trend_strength_col = f'trend_strength_{window}d'

        def calc_trend_r2(series):
            if len(series) < window:
                return np.nan
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation ** 2

        new_features[trend_strength_col] = df.groupby('symbol')[price_col].transform(
            lambda x: x.rolling(window, min_periods=max(5, window//2)).apply(calc_trend_r2, raw=False)
        )
        print(f"  Created: {trend_strength_col}")

        # Trend slope (rate of change)
        trend_slope_col = f'trend_slope_{window}d'

        def calc_slope(series):
            if len(series) < window:
                return np.nan
            x = np.arange(len(series))
            y = series.values
            if np.std(y) == 0:
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope / np.mean(y) if np.mean(y) != 0 else 0  # Normalized slope

        new_features[trend_slope_col] = df.groupby('symbol')[price_col].transform(
            lambda x: x.rolling(window, min_periods=max(5, window//2)).apply(calc_slope, raw=False)
        )
        print(f"  Created: {trend_slope_col}")

    print(f"\nCreated {len(new_features.columns)} trend features")

    return new_features


def create_frequency_domain_features(df, price_col='close', n_components=3):
    """
    Create frequency domain features using FFT.

    Extract dominant frequency components that might capture cyclical patterns.
    """
    print("\n" + "=" * 80)
    print("CREATING FREQUENCY DOMAIN FEATURES")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    window = 20  # Use 20-day window for FFT

    print(f"\nUsing {window}-day FFT window")
    print(f"Extracting top {n_components} frequency components")

    for comp in range(n_components):
        fft_col = f'fft_component_{comp+1}'

        def calc_fft_component(series):
            if len(series) < window:
                return np.nan
            # Detrend
            detrended = series - np.linspace(series.iloc[0], series.iloc[-1], len(series))
            # FFT
            fft_vals = fft(detrended.values)
            # Get magnitude of component (skip DC component at 0)
            if comp + 1 < len(fft_vals):
                return np.abs(fft_vals[comp + 1])
            return np.nan

        new_features[fft_col] = df.groupby('symbol')[price_col].transform(
            lambda x: x.rolling(window, min_periods=window).apply(calc_fft_component, raw=False)
        )
        print(f"  Created: {fft_col}")

    print(f"\nCreated {n_components} FFT features")

    return new_features


def create_nonlinear_transforms(df, feature_list):
    """
    Create non-linear transformations of important features.

    Transforms:
    - Log of absolute values (for returns)
    - Square root (for volatility-like features)
    - Squared terms (for capturing non-linear effects)
    """
    print("\n" + "=" * 80)
    print("CREATING NON-LINEAR TRANSFORMATIONS")
    print("=" * 80)

    new_features = pd.DataFrame(index=df.index)

    # Select features suitable for non-linear transforms
    transform_candidates = [
        ('returns_1d', 'squared'),
        ('volatility_20d', 'sqrt'),
        ('volume_ratio_5d', 'log'),
        ('vix_change_1d_y', 'squared'),
        ('spy_returns_5d_y', 'squared'),
    ]

    for feat, transform_type in transform_candidates:
        if feat not in df.columns:
            continue

        if transform_type == 'squared':
            new_col = f'{feat}_squared'
            new_features[new_col] = df[feat] ** 2
            print(f"  Created: {new_col}")

        elif transform_type == 'sqrt':
            new_col = f'{feat}_sqrt'
            # Ensure non-negative for sqrt
            new_features[new_col] = np.sqrt(np.abs(df[feat]))
            print(f"  Created: {new_col}")

        elif transform_type == 'log':
            new_col = f'{feat}_log'
            # Add small constant to avoid log(0)
            new_features[new_col] = np.log1p(np.abs(df[feat]))
            print(f"  Created: {new_col}")

    print(f"\nCreated {len(new_features.columns)} non-linear transform features")

    return new_features


def main():
    """Create all advanced features and save enhanced dataset."""

    print("\n" + "=" * 80)
    print("ADVANCED FEATURE GENERATION")
    print("=" * 80)
    print("\nGoal: Add 20-30 high-quality NEW features")
    print("Target: 59.12% -> 60%+ accuracy")
    print("\nFeature types:")
    print("  1. Feature Interactions (top features crossed)")
    print("  2. Higher-order Statistics (skewness, kurtosis)")
    print("  3. Trend Features (trend strength, slope)")
    print("  4. Frequency Domain (FFT components)")
    print("  5. Non-linear Transformations")

    # Load existing dataset with 60 selected features
    data_dir = Path(__file__).parent.parent / "data"
    df = pd.read_parquet(data_dir / "stocks_selected_features.parquet")

    print(f"\n{'-' * 80}")
    print("LOADING BASE DATASET")
    print(f"{'-' * 80}")
    print(f"\nOriginal shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get list of existing features
    meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                 'future_return', 'label']
    existing_features = [c for c in df.columns if c not in meta_cols]
    print(f"Existing features: {len(existing_features)}")

    # Top features for interactions (based on previous importance analysis)
    top_features = [
        'uso_returns_1d_y', 'dpo_20', 'vix_change_1d_y', 'tlt_returns_5d_y',
        'gld_returns_1d_y', 'qqq_returns_1d_y', 'uup_returns_1d_y', 'tlt_returns_1d_y',
        'vix_level_y', 'qqq_returns_5d_y', 'gld_returns_5d_y', 'vix_change_5d_y',
        'iwm_returns_1d_y', 'dia_returns_5d_y', 'spy_returns_20d_y', 'mfi_14',
        'volume_ratio_20d', 'market_breadth_1d'
    ]

    # Create each type of advanced feature
    all_new_features = []

    # 1. Feature Interactions
    interactions = create_feature_interactions(df, top_features, max_interactions=15)
    all_new_features.append(interactions)

    # 2. Higher-order Statistics
    higher_stats = create_higher_order_stats(df, windows=[5, 20])
    all_new_features.append(higher_stats)

    # 3. Trend Features
    trend_feats = create_trend_features(df, windows=[10, 20])
    all_new_features.append(trend_feats)

    # 4. Frequency Domain
    fft_feats = create_frequency_domain_features(df, n_components=3)
    all_new_features.append(fft_feats)

    # 5. Non-linear Transformations
    nonlinear_feats = create_nonlinear_transforms(df, existing_features)
    all_new_features.append(nonlinear_feats)

    # Combine all new features
    print("\n" + "=" * 80)
    print("COMBINING FEATURES")
    print("=" * 80)

    df_new_features = pd.concat(all_new_features, axis=1)

    # Remove columns with too many NaNs
    nan_threshold = 0.5  # Remove if >50% NaN
    nan_pct = df_new_features.isna().mean()
    valid_cols = nan_pct[nan_pct < nan_threshold].index.tolist()

    print(f"\nNew features created: {len(df_new_features.columns)}")
    print(f"Valid features (< 50% NaN): {len(valid_cols)}")
    print(f"Removed features (too many NaN): {len(df_new_features.columns) - len(valid_cols)}")

    df_new_features = df_new_features[valid_cols]

    # Combine with original dataset
    df_enhanced = pd.concat([df, df_new_features], axis=1)

    print(f"\nEnhanced dataset shape: {df_enhanced.shape}")
    print(f"Original features: {len(existing_features)}")
    print(f"New features: {len(valid_cols)}")
    print(f"Total features: {len(existing_features) + len(valid_cols)}")

    # Save enhanced dataset
    output_path = data_dir / "stocks_enhanced_features.parquet"
    df_enhanced.to_parquet(output_path, index=False)

    print(f"\n{'-' * 80}")
    print(f"Enhanced dataset saved to: {output_path}")
    print(f"{'-' * 80}")

    # Show new feature summary
    print(f"\nNew feature categories:")
    print(f"  - Interactions: {len([c for c in valid_cols if 'interaction_' in c])}")
    print(f"  - Higher-order stats: {len([c for c in valid_cols if any(x in c for x in ['skew', 'kurtosis', 'iqr'])])}")
    print(f"  - Trend features: {len([c for c in valid_cols if 'trend_' in c])}")
    print(f"  - FFT features: {len([c for c in valid_cols if 'fft_' in c])}")
    print(f"  - Non-linear: {len([c for c in valid_cols if any(x in c for x in ['squared', 'sqrt', 'log'])])}")

    print(f"\n{'=' * 80}")
    print("FEATURE GENERATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nNext step: Train with enhanced features")
    print(f"  python train/train_with_enhanced_features.py")
    print(f"\nExpected: 59.12% -> 59.5-60.2% (+0.4-1.0%)")


if __name__ == "__main__":
    main()
