"""
Update stocks_selected_features.parquet from stocks_features.parquet.

This script:
1. Loads stocks_features.parquet (enriched output of prepare_prediction_data;
   stocks.parquet itself stays RAW as of the P1 pipeline separation)
2. Renames market features to add '_y' suffix (to match training data format)
3. Selects the 60 features used by the model
4. Saves as stocks_selected_features.parquet
"""

import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent / "data"
df = pd.read_parquet(data_dir / "stocks_features.parquet")

print("=" * 80)
print("UPDATING STOCKS_SELECTED_FEATURES.PARQUET")
print("=" * 80)
print(f"\nLoaded stocks_features.parquet:")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Shape: {df.shape}")

# Market features that need '_y' suffix
market_features_to_rename = {
    'dia_returns_1d': 'dia_returns_1d_y',
    'dia_returns_5d': 'dia_returns_5d_y',
    'gld_returns_1d': 'gld_returns_1d_y',
    'gld_returns_5d': 'gld_returns_5d_y',
    'iwm_returns_1d': 'iwm_returns_1d_y',
    'iwm_returns_5d': 'iwm_returns_5d_y',
    'qqq_returns_1d': 'qqq_returns_1d_y',
    'qqq_returns_5d': 'qqq_returns_5d_y',
    'spy_returns_1d': 'spy_returns_1d_y',
    'spy_returns_5d': 'spy_returns_5d_y',
    'spy_returns_20d': 'spy_returns_20d_y',
    'spy_volatility_20d': 'spy_volatility_20d_y',
    'tlt_returns_1d': 'tlt_returns_1d_y',
    'tlt_returns_5d': 'tlt_returns_5d_y',
    'uso_returns_1d': 'uso_returns_1d_y',
    'uso_returns_5d': 'uso_returns_5d_y',
    'uup_returns_1d': 'uup_returns_1d_y',
    'uup_returns_5d': 'uup_returns_5d_y',
    'vix_level': 'vix_level_y',
    'vix_change_1d': 'vix_change_1d_y',
    'vix_change_5d': 'vix_change_5d_y',
}

# Rename market features. If a '_y' target already exists (stale column from
# an older enrichment cycle), drop it first so the rename can never produce
# duplicate column names.
stale = [dst for src, dst in market_features_to_rename.items()
         if src in df.columns and dst in df.columns]
if stale:
    print(f"\n[WARN] Dropping {len(stale)} stale '_y' columns from a previous cycle")
    df = df.drop(columns=stale)
print(f"\nRenaming {len(market_features_to_rename)} market features to add '_y' suffix...")
df = df.rename(columns=market_features_to_rename)
assert not df.columns.duplicated().any(), "duplicate columns after rename"

# The 60 features used by the model (excluding meta columns)
selected_features = [
    'adx_strong_trend',
    'alpha_019',
    'alpha_041',
    'alpha_042',
    'bb_squeeze',
    'bb_width',
    'cci_20',
    'consecutive_down_days',
    'consecutive_up_days',
    'dia_returns_1d_y',
    'dia_returns_5d_y',
    'dpo_20',
    'gld_returns_1d_y',
    'gld_returns_5d_y',
    'golden_cross',
    'iwm_returns_1d_y',
    'iwm_returns_5d_y',
    'kc_position',
    'macd',
    'macd_hist',
    'macd_positive',
    'mfi_14',
    'momentum_50d',
    'price_above_ema_50',
    'price_above_sma_20',
    'psar',
    'qqq_returns_1d_y',
    'qqq_returns_5d_y',
    'returns_10d',
    'returns_1d_positive',
    'returns_20d',
    'returns_2d',
    'returns_5d',
    'returns_60d',
    'rsi_7',
    'rsi_oversold',
    'spy_returns_1d_y',
    'spy_returns_20d_y',
    'spy_returns_5d_y',
    'spy_volatility_20d_y',
    'supertrend',
    'tlt_returns_1d_y',
    'tlt_returns_5d_y',
    'trix_14',
    'uso_returns_1d_y',
    'uso_returns_5d_y',
    'uup_returns_1d_y',
    'uup_returns_5d_y',
    'vix_change_1d_y',
    'vix_change_5d_y',
    'vix_level_y',
    'volatility_10d',
    'volatility_20d',
    'volatility_60d',
    'volume_increasing',
    'volume_ratio_20d',
    'volume_ratio_5d',
    'volume_std_20d',
    'vwap_ratio',
    'williams_r_14',
]

# Meta columns to keep
meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']

# Check which selected features are missing
missing = [f for f in selected_features if f not in df.columns]
if missing:
    print(f"\n[WARNING] Missing {len(missing)} features:")
    for f in missing:
        print(f"  - {f}")

# Keep only selected features + meta columns
available_features = [f for f in selected_features if f in df.columns]
cols_to_keep = meta_cols + available_features

print(f"\nSelecting {len(available_features)} features + {len(meta_cols)} meta columns...")
df_selected = df[cols_to_keep].copy()

# Save
output_path = data_dir / "stocks_selected_features.parquet"
df_selected.to_parquet(output_path, index=False, compression='snappy')

print(f"\n{'=' * 80}")
print("SUCCESS")
print(f"{'=' * 80}")
print(f"\nSaved to: {output_path}")
print(f"  Date range: {df_selected['date'].min()} to {df_selected['date'].max()}")
print(f"  Shape: {df_selected.shape}")
print(f"  Features: {len(available_features)}")

if missing:
    print(f"\n[NOTE] {len(missing)} features were missing from source data.")
    print("This may affect model performance.")
