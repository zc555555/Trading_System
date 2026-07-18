"""
Fetch Option Implied Volatility (IV) data for stocks.

Data source: Yahoo Finance (free)
Extracts IV from option chains for each stock.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import yaml

def get_iv_from_options(ticker, date):
    """
    Get implied volatility from option chains.

    Args:
        ticker: Stock symbol
        date: Date to fetch IV for

    Returns:
        Average IV across near-the-money options, or None if unavailable
    """
    try:
        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = stock.options
        if len(expirations) == 0:
            return None

        # Use nearest expiration (30-60 days out preferred)
        exp_date = expirations[0]

        # Get option chain
        opt_chain = stock.option_chain(exp_date)

        # Get current stock price
        current_price = stock.history(period='1d')['Close'].iloc[-1]

        # Filter near-the-money options (within 5% of current price)
        calls = opt_chain.calls
        calls_ntm = calls[
            (calls['strike'] >= current_price * 0.95) &
            (calls['strike'] <= current_price * 1.05)
        ]

        if len(calls_ntm) == 0:
            return None

        # Average implied volatility
        avg_iv = calls_ntm['impliedVolatility'].mean()

        # Convert to percentage
        return avg_iv * 100 if not np.isnan(avg_iv) else None

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        return None


def fetch_historical_iv(symbols, start_date, end_date):
    """
    Fetch historical IV data for multiple stocks.

    Note: Yahoo Finance doesn't provide historical IV directly.
    We'll fetch current IV and VIX as proxy for historical volatility regime.

    For production: Use paid services like CBOE or OptionMetrics.
    For now: Use VIX + current IV as approximation.
    """

    print("=" * 80)
    print("FETCHING OPTION IMPLIED VOLATILITY DATA")
    print("=" * 80)

    print("\nNote: Yahoo Finance limits historical IV access.")
    print("Strategy: Use VIX (market IV) + current stock IV")

    # Fetch VIX (market-wide implied volatility)
    print("\nFetching VIX (market IV proxy)...")
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix['Close']
    vix.name = 'vix'

    print(f"  VIX data: {len(vix)} days")

    # For each stock, fetch current IV and calculate historical proxy
    print(f"\nFetching IV data for {len(symbols)} stocks...")

    iv_data = {}

    for symbol in symbols:
        print(f"\n  Processing {symbol}...")

        try:
            # Get stock historical volatility (realized vol, proxy for IV)
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            if len(hist) == 0:
                print(f"    [SKIP] No price data")
                continue

            # Calculate realized volatility (20-day rolling)
            returns = hist['Close'].pct_change()
            realized_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100

            # Get current IV
            current_iv = get_iv_from_options(symbol, None)

            if current_iv is not None:
                print(f"    Current IV: {current_iv:.2f}%")

                # Adjust realized vol to match current IV level
                # (Scale factor to align historical realized vol with current IV)
                if len(realized_vol.dropna()) > 0:
                    recent_realized = realized_vol.iloc[-20:].mean()
                    if recent_realized > 0:
                        scale_factor = current_iv / recent_realized
                        iv_proxy = realized_vol * scale_factor
                    else:
                        iv_proxy = realized_vol
                else:
                    iv_proxy = realized_vol
            else:
                print(f"    Using realized volatility as IV proxy")
                iv_proxy = realized_vol

            iv_data[symbol] = iv_proxy
            print(f"    Collected {len(iv_proxy.dropna())} days of IV data")

        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

    # Combine into DataFrame
    print("\n" + "=" * 80)
    print("CREATING IV DATASET")
    print("=" * 80)

    # Create multi-index dataframe
    all_data = []

    for symbol, iv_series in iv_data.items():
        df = pd.DataFrame({
            'date': iv_series.index,
            'symbol': symbol,
            'implied_volatility': iv_series.values
        })
        all_data.append(df)

    if len(all_data) == 0:
        print("\n[ERROR] No IV data collected!")
        return None

    df_iv = pd.concat(all_data, ignore_index=True)

    # Add VIX for market-wide IV context
    vix_df = vix.reset_index()
    vix_df.columns = ['date', 'vix']  # Explicitly set column names
    # Remove timezone info to match stock data
    if hasattr(vix_df['date'].dtype, 'tz') and vix_df['date'].dtype.tz is not None:
        vix_df['date'] = vix_df['date'].dt.tz_localize(None)
    if hasattr(df_iv['date'].dtype, 'tz') and df_iv['date'].dtype.tz is not None:
        df_iv['date'] = df_iv['date'].dt.tz_localize(None)

    df_iv = df_iv.merge(vix_df, on='date', how='left')

    print(f"\nFinal IV dataset: {df_iv.shape}")
    print(f"  Symbols: {df_iv['symbol'].nunique()}")
    print(f"  Date range: {df_iv['date'].min()} to {df_iv['date'].max()}")
    print(f"  Total records: {len(df_iv):,}")

    # Save
    output_dir = Path(__file__).parent
    output_path = output_dir / "option_iv.parquet"

    df_iv.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df_iv


def main():
    """Main execution."""

    # Load config (resolves named universe → symbols if needed)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import load_config
    config = load_config()
    symbols = config['data']['symbols']
    start_date = config['data'].get('start_date', '2018-01-01')

    print(f"\nSymbols: {len(symbols)}")
    print(f"Start date: {start_date}")

    # Fetch IV data
    df_iv = fetch_historical_iv(symbols, start_date, datetime.now().strftime('%Y-%m-%d'))

    if df_iv is not None:
        print("\n" + "=" * 80)
        print("IV DATA COLLECTION COMPLETE!")
        print("=" * 80)
        print("\nNext step: Process IV sequences with LSTM")
        print("  python features/process_iv_with_lstm.py")


if __name__ == "__main__":
    main()
