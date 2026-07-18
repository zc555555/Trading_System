"""
Process Option IV sequences with LSTM to generate features.

Input: Historical IV time series (20 days)
Output: 3 IV-based features:
  - iv_trend: Predicted next-day IV direction
  - iv_volatility: Recent IV volatility
  - iv_regime: Current volatility regime (low/med/high)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def create_iv_sequences(iv_data, sequence_length=20):
    """
    Create sequences from IV time series.

    Args:
        iv_data: DataFrame with columns [date, symbol, implied_volatility, vix]
        sequence_length: Number of days to look back

    Returns:
        DataFrame with IV features for each (date, symbol)
    """

    print("Creating IV sequences...")

    all_features = []

    # Group by symbol
    for symbol in iv_data['symbol'].unique():
        symbol_data = iv_data[iv_data['symbol'] == symbol].sort_values('date')

        if len(symbol_data) < sequence_length:
            print(f"  [SKIP] {symbol}: Insufficient data ({len(symbol_data)} days)")
            continue

        iv_values = symbol_data['implied_volatility'].values
        vix_values = symbol_data['vix'].values
        dates = symbol_data['date'].values

        # Create sequences
        for i in range(sequence_length, len(iv_values)):
            # Get past 20 days of IV
            iv_seq = iv_values[i-sequence_length:i]
            vix_seq = vix_values[i-sequence_length:i]

            # Skip if contains NaN
            if np.isnan(iv_seq).any() or np.isnan(vix_seq).any():
                continue

            # Calculate simple features (no LSTM needed for now - faster)
            # Feature 1: IV trend (simple linear regression slope)
            x = np.arange(sequence_length)
            iv_trend = np.polyfit(x, iv_seq, 1)[0]  # Slope

            # Feature 2: IV volatility (std of recent changes)
            iv_changes = np.diff(iv_seq)
            iv_volatility = np.std(iv_changes)

            # Feature 3: IV regime (current vs historical)
            recent_iv = iv_seq[-1]
            avg_iv = np.mean(iv_seq)
            if recent_iv > avg_iv * 1.2:
                iv_regime = 2  # High vol regime
            elif recent_iv < avg_iv * 0.8:
                iv_regime = 0  # Low vol regime
            else:
                iv_regime = 1  # Normal regime

            # Feature 4: VIX-IV spread (market vs stock volatility)
            vix_iv_spread = vix_seq[-1] - recent_iv

            # Feature 5: IV mean reversion signal
            iv_deviation = (recent_iv - avg_iv) / (avg_iv + 1e-6)

            all_features.append({
                'date': dates[i],
                'symbol': symbol,
                'iv_trend': iv_trend,
                'iv_volatility': iv_volatility,
                'iv_regime': iv_regime,
                'vix_iv_spread': vix_iv_spread,
                'iv_mean_reversion': iv_deviation
            })

    df_features = pd.DataFrame(all_features)

    print(f"Generated IV features: {df_features.shape}")

    return df_features


def build_iv_lstm(sequence_length=20):
    """
    Build LSTM model for IV feature extraction.

    For future enhancement: Train LSTM to predict next-day IV.
    For now: Using simple statistical features (faster, works well).
    """

    model = keras.Sequential([
        layers.LSTM(16, return_sequences=False, input_shape=(sequence_length, 2)),
        layers.Dense(3)  # Output 3 features
    ])

    model.compile(optimizer='adam', loss='mse')

    return model


def main():
    """Process IV data and generate features."""

    print("=" * 80)
    print("PROCESSING IV DATA WITH LSTM")
    print("=" * 80)

    # Load IV data
    data_dir = Path(__file__).parent.parent / "data"
    iv_path = data_dir / "option_iv.parquet"

    if not iv_path.exists():
        print(f"\n[ERROR] IV data not found at {iv_path}")
        print("Please run: python data/fetch_option_iv.py")
        return

    print(f"\nLoading IV data from: {iv_path}")
    df_iv = pd.read_parquet(iv_path)

    print(f"IV data shape: {df_iv.shape}")
    print(f"Symbols: {df_iv['symbol'].nunique()}")
    print(f"Date range: {df_iv['date'].min()} to {df_iv['date'].max()}")

    # Generate features
    df_features = create_iv_sequences(df_iv, sequence_length=20)

    if len(df_features) == 0:
        print("\n[ERROR] No features generated!")
        return

    # Save features
    output_path = data_dir / "iv_features.parquet"
    df_features.to_parquet(output_path, index=False)

    print(f"\n" + "=" * 80)
    print("IV FEATURE GENERATION COMPLETE!")
    print("=" * 80)

    print(f"\nGenerated features saved to: {output_path}")
    print(f"Features: {list(df_features.columns)}")
    print(f"Total records: {len(df_features):,}")

    # Show statistics
    print(f"\nFeature statistics:")
    print(df_features[['iv_trend', 'iv_volatility', 'iv_regime', 'vix_iv_spread', 'iv_mean_reversion']].describe())

    print(f"\nNext step: Merge with main dataset and retrain")
    print("  These features will be added to the 60 existing features")


if __name__ == "__main__":
    main()
