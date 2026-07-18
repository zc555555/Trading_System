"""
Data fetcher for stock OHLCV data using yfinance.
Handles data downloading, cleaning, and storage in Parquet format.

W1 (2026-05) upgrades:
- Resolves `data.symbols` config field. If it is null/empty, the universe
  is resolved from `data.universe` via research.data.universe.get_universe().
- Logs per-symbol failures to a manifest so we can audit the universe
  after a 500-symbol fetch.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Allow `python research/data/fetch_ohlcv.py` and `from research.data.fetch_ohlcv ...`
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    from data.universe import get_universe
except ImportError:
    # Fallback when imported as a submodule
    from .universe import get_universe  # type: ignore


class StockDataFetcher:
    """Fetches and manages stock OHLCV data."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data fetcher.

        Args:
            data_dir: Directory to store parquet files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            adjust: Whether to use adjusted prices (recommended: True)

        Returns:
            DataFrame with MultiIndex (date, symbol) and columns [open, high, low, close, volume]
        """
        if end_date is None:
            # Add 1 day because yfinance end date is exclusive
            from datetime import timedelta
            tomorrow = datetime.now() + timedelta(days=1)
            end_date = tomorrow.strftime('%Y-%m-%d')

        all_data = []

        print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}...")

        for symbol in tqdm(symbols, desc="Downloading"):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=adjust)

                if df.empty:
                    print(f"Warning: No data found for {symbol}")
                    continue

                # Rename columns to lowercase
                df.columns = df.columns.str.lower()

                # Keep only OHLCV
                df = df[['open', 'high', 'low', 'close', 'volume']]

                # Add symbol column
                df['symbol'] = symbol

                # Reset index to make date a column
                df = df.reset_index()
                df.columns = df.columns.str.lower()

                all_data.append(df)

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data was successfully fetched for any symbol")

        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        # Sort by date and symbol
        combined = combined.sort_values(['date', 'symbol']).reset_index(drop=True)

        print(f"\nFetched {len(combined):,} rows across {len(symbols)} symbols")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

        return combined

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data: handle missing values, outliers, etc.

        Args:
            df: Raw OHLCV dataframe

        Returns:
            Cleaned dataframe
        """
        df = df.copy()

        # Remove rows with missing OHLC values
        initial_rows = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} rows with missing OHLC values")

        # Fill missing volume with 0
        df['volume'] = df['volume'].fillna(0)

        # Detect and handle obvious data errors
        # (e.g., high < low, close outside [low, high])
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )

        if invalid_mask.sum() > 0:
            print(f"Warning: Found {invalid_mask.sum()} rows with invalid OHLC relationships")
            print("Removing these rows...")
            df = df[~invalid_mask]

        # Detect unrealistic price changes (>50% in one day - likely split not adjusted)
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        extreme_returns = df['returns'].abs() > 0.5

        if extreme_returns.sum() > 0:
            print(f"Warning: Found {extreme_returns.sum()} extreme returns (>50% daily change)")
            print("These might be unadjusted splits - review manually if needed")

        df = df.drop(columns=['returns'])

        return df

    def save_parquet(self, df: pd.DataFrame, filename: str = "stocks.parquet"):
        """
        Save dataframe to parquet file.

        Args:
            df: Dataframe to save
            filename: Output filename
        """
        output_path = self.data_dir / filename
        df.to_parquet(output_path, index=False, compression='snappy')

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved to {output_path} ({file_size_mb:.2f} MB)")

    def load_parquet(self, filename: str = "stocks.parquet") -> pd.DataFrame:
        """
        Load dataframe from parquet file.

        Args:
            filename: Input filename

        Returns:
            Loaded dataframe
        """
        input_path = self.data_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        df = pd.read_parquet(input_path)
        print(f"Loaded {len(df):,} rows from {input_path}")

        return df


def resolve_symbols_from_config(config: dict) -> List[str]:
    """Resolve the equity universe from a parsed config dict.

    Order of precedence:
      1. ``data.symbols`` if it is a non-empty list (legacy override / rollback).
      2. ``data.universe`` named universe via universe.get_universe.

    ETF benchmarks are appended when ``data.include_benchmark_etfs`` is true.
    """
    data_cfg = config.get('data', {})

    explicit = data_cfg.get('symbols')
    if explicit:  # non-empty list overrides
        print(f"[fetch] Using explicit symbol list ({len(explicit)} tickers) from config")
        return list(explicit)

    universe_name = data_cfg.get('universe', 'sp100')
    include_benchmarks = bool(data_cfg.get('include_benchmark_etfs', False))
    print(f"[fetch] Resolving universe '{universe_name}' "
          f"(include_benchmark_etfs={include_benchmarks})")
    symbols = get_universe(universe_name, include_benchmarks=include_benchmarks)
    print(f"[fetch] Universe resolved to {len(symbols)} tickers")
    return symbols


def main():
    """Main function for command-line usage."""
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve universe
    symbols = resolve_symbols_from_config(config)
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    output_file = config['data']['parquet_path']

    # Fetch data
    fetcher = StockDataFetcher()

    print("=" * 60)
    print("Stock Data Fetcher")
    print("=" * 60)
    print(f"Universe size: {len(symbols)} tickers")
    print(f"Date range: {start_date} -> {end_date or 'today'}")
    print(f"Output: data/{output_file}")

    df = fetcher.fetch_symbols(symbols, start_date, end_date, adjust=True)

    print("\n" + "=" * 60)
    print("Cleaning data...")
    print("=" * 60)

    df = fetcher.clean_data(df)

    print("\n" + "=" * 60)
    print("Saving data...")
    print("=" * 60)

    fetcher.save_parquet(df, output_file)

    # Save a manifest of which symbols were successfully fetched
    fetched_symbols = sorted(df['symbol'].unique().tolist())
    requested = set(symbols)
    missing = sorted(requested - set(fetched_symbols))
    manifest_path = Path(__file__).parent / "fetch_manifest.json"
    import json
    with open(manifest_path, 'w') as f:
        json.dump({
            "requested_count": len(symbols),
            "fetched_count": len(fetched_symbols),
            "missing_count": len(missing),
            "fetched": fetched_symbols,
            "missing": missing,
            "fetched_at": datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"Requested: {len(symbols)}  |  Fetched: {len(fetched_symbols)}  |  Missing: {len(missing)}")
    if missing:
        preview = missing[:20]
        more = "" if len(missing) <= 20 else f"  (+{len(missing)-20} more)"
        print(f"Missing tickers: {preview}{more}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total rows: {len(df):,}")
    print(f"Manifest saved to: {manifest_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
