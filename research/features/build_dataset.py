"""
Build machine learning dataset from raw OHLCV data.

This module:
1. Loads raw data from parquet
2. Computes alpha features
3. Generates labels (future returns / signals)
4. Handles missing values
5. Exports dataset + feature manifest
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
import warnings

import numpy as np
import pandas as pd
import yaml

# Handle both direct execution and module import
try:
    from .alphas_101_subset import compute_all_alphas, ALPHAS
except ImportError:
    from alphas_101_subset import compute_all_alphas, ALPHAS


class DatasetBuilder:
    """Build ML-ready dataset from OHLCV data."""

    def __init__(self, config_path: str = None):
        """
        Initialize dataset builder.

        Args:
            config_path: Path to config.yaml (default: ../config.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.max_lookback = self.config['features']['max_lookback']
        self.label_config = self.config['label']

    def load_data(self, parquet_path: str = None) -> pd.DataFrame:
        """
        Load raw OHLCV data.

        Args:
            parquet_path: Path to parquet file

        Returns:
            DataFrame with OHLCV data
        """
        if parquet_path is None:
            parquet_path = Path(__file__).parent.parent / self.config['data']['parquet_path']

        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        # Ensure correct dtypes
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

        print(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def compute_features(self, df: pd.DataFrame, alpha_list: List[str] = None) -> pd.DataFrame:
        """
        Compute alpha features.

        Args:
            df: Raw OHLCV data
            alpha_list: List of alpha names to compute (default: from config)

        Returns:
            DataFrame with original data + features
        """
        if alpha_list is None:
            # Use alphas from config or all available
            if 'alphas_101' in self.config['features'] and self.config['features']['alphas_101']:
                alpha_list = [f"alpha_{str(i).zfill(3)}" for i in self.config['features']['alphas_101']]
            elif 'alphas_101' in self.config['features'] and not self.config['features']['alphas_101']:
                # Empty list explicitly provided - skip alphas
                alpha_list = []
            else:
                alpha_list = list(ALPHAS.keys())

        if len(alpha_list) > 0:
            print(f"\n{'=' * 60}")
            print(f"Computing {len(alpha_list)} alpha features...")
            print(f"{'=' * 60}\n")

            # Filter to only available alphas
            available_alphas = [a for a in alpha_list if a in ALPHAS]

            if len(available_alphas) < len(alpha_list):
                missing = set(alpha_list) - set(available_alphas)
                print(f"Warning: {len(missing)} alphas not found: {missing}")

            # Compute alphas
            df_features = compute_all_alphas(df, available_alphas)
        else:
            print(f"\n{'=' * 60}")
            print(f"Skipping alpha features (single stock mode)")
            print(f"{'=' * 60}\n")
            df_features = df

        # Add basic technical features if specified
        if 'technical' in self.config['features']:
            df_features = self._add_technical_features(df_features)

        if len(alpha_list) > 0:
            print(f"\nTotal alpha features computed: {len(available_alphas)}")

        return df_features

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators beyond alphas."""
        try:
            from . import operators as ops
        except ImportError:
            import operators as ops

        print("\nAdding technical features...")

        for tech in self.config['features']['technical']:
            try:
                # Returns features
                if tech == 'returns_1d':
                    df[tech] = ops.returns(df, 'close', 1)
                elif tech == 'returns_2d':
                    df[tech] = ops.returns(df, 'close', 2)
                elif tech == 'returns_5d':
                    df[tech] = ops.returns(df, 'close', 5)
                elif tech == 'returns_10d':
                    df[tech] = ops.returns(df, 'close', 10)
                elif tech == 'returns_20d':
                    df[tech] = ops.returns(df, 'close', 20)

                # Volatility features
                elif tech == 'volatility_5d':
                    df[tech] = ops.volatility(df, 5, 'close')
                elif tech == 'volatility_10d':
                    df[tech] = ops.volatility(df, 10, 'close')
                elif tech == 'volatility_20d':
                    df[tech] = ops.volatility(df, 20, 'close')

                # Volume features
                elif tech == 'volume_ratio_5d':
                    vol_ma = ops.ts_mean(df, 'volume', 5)
                    df[tech] = df['volume'] / vol_ma
                elif tech == 'volume_ratio_20d':
                    vol_ma = ops.ts_mean(df, 'volume', 20)
                    df[tech] = df['volume'] / vol_ma

                # RSI
                elif tech == 'rsi_7':
                    df[tech] = ops.rsi(df, 'close', 7)
                elif tech == 'rsi_14':
                    df[tech] = ops.rsi(df, 'close', 14)
                elif tech == 'rsi_21':
                    df[tech] = ops.rsi(df, 'close', 21)

                # MACD
                elif tech == 'macd':
                    macd, signal, hist = ops.macd_indicator(df, 'close')
                    df[tech] = macd
                elif tech == 'macd_signal':
                    macd, signal, hist = ops.macd_indicator(df, 'close')
                    df[tech] = signal
                elif tech == 'macd_hist':
                    macd, signal, hist = ops.macd_indicator(df, 'close')
                    df[tech] = hist

                # Bollinger Bands
                elif tech == 'bb_upper':
                    upper, lower, pos = ops.bollinger_bands(df, 'close', 20)
                    df[tech] = upper
                elif tech == 'bb_lower':
                    upper, lower, pos = ops.bollinger_bands(df, 'close', 20)
                    df[tech] = lower
                elif tech == 'bb_position':
                    upper, lower, pos = ops.bollinger_bands(df, 'close', 20)
                    df[tech] = pos

                # EMA
                elif tech == 'ema_5':
                    df[tech] = ops.ema(df, 'close', 5)
                elif tech == 'ema_10':
                    df[tech] = ops.ema(df, 'close', 10)
                elif tech == 'ema_20':
                    df[tech] = ops.ema(df, 'close', 20)

                # Price to EMA ratio
                elif tech == 'price_to_ema20':
                    ema20 = ops.ema(df, 'close', 20)
                    df[tech] = df['close'] / ema20

                # ATR (Average True Range)
                elif tech == 'atr_14':
                    df[tech] = ops.atr(df, 14)

                # ADX (Average Directional Index)
                elif tech == 'adx_14':
                    df[tech] = ops.adx(df, 14)

                # Additional returns
                elif tech == 'returns_60d':
                    df[tech] = ops.returns(df, 'close', 60)

                # Additional volatility
                elif tech == 'volatility_60d':
                    df[tech] = ops.volatility(df, 60, 'close')

                # Volume std
                elif tech == 'volume_std_20d':
                    df[tech] = ops.volume_std(df, 20)

                # Additional RSI
                elif tech == 'rsi_50':
                    df[tech] = ops.rsi(df, 'close', 50)

                # BB width
                elif tech == 'bb_width':
                    df[tech] = ops.bb_width(df, 'close', 20)

                # Additional EMA
                elif tech == 'ema_50':
                    df[tech] = ops.ema(df, 'close', 50)
                elif tech == 'ema_200':
                    df[tech] = ops.ema(df, 'close', 200)

                # SMA
                elif tech == 'sma_10':
                    df[tech] = ops.sma(df, 'close', 10)
                elif tech == 'sma_20':
                    df[tech] = ops.sma(df, 'close', 20)
                elif tech == 'sma_50':
                    df[tech] = ops.sma(df, 'close', 50)
                elif tech == 'sma_200':
                    df[tech] = ops.sma(df, 'close', 200)

                # Stochastic Oscillator
                elif tech == 'stochastic_k':
                    stoch_k, stoch_d = ops.stochastic_oscillator(df, 14, 3, 3)
                    df['stochastic_k'] = stoch_k
                    df['stochastic_d'] = stoch_d
                    print(f"[OK] stochastic_k and stochastic_d")
                    continue
                elif tech == 'stochastic_d':
                    if 'stochastic_d' not in df.columns:
                        stoch_k, stoch_d = ops.stochastic_oscillator(df, 14, 3, 3)
                        df['stochastic_d'] = stoch_d

                # CCI
                elif tech == 'cci_20':
                    df[tech] = ops.cci(df, 20)

                # Williams %R
                elif tech == 'williams_r_14':
                    df[tech] = ops.williams_r(df, 14)

                # OBV
                elif tech == 'obv':
                    df[tech] = ops.obv(df)
                elif tech == 'obv_ema_20':
                    if 'obv' not in df.columns:
                        df['obv'] = ops.obv(df)
                    df[tech] = ops.ema(df, 'obv', 20)

                # CMF
                elif tech == 'cmf_20':
                    df[tech] = ops.cmf(df, 20)

                # MFI
                elif tech == 'mfi_14':
                    df[tech] = ops.mfi(df, 14)

                # TRIX
                elif tech == 'trix_14':
                    df[tech] = ops.trix(df, 'close', 14)

                # DPO
                elif tech == 'dpo_20':
                    df[tech] = ops.dpo(df, 'close', 20)

                # Keltner Channel
                elif tech == 'kc_upper':
                    kc_upper, kc_lower, kc_pos = ops.keltner_channel(df, 'close', 20, 2.0)
                    df['kc_upper'] = kc_upper
                    df['kc_lower'] = kc_lower
                    df['kc_position'] = kc_pos
                    print(f"[OK] kc_upper, kc_lower, kc_position")
                    continue
                elif tech == 'kc_lower':
                    if 'kc_lower' not in df.columns:
                        kc_upper, kc_lower, kc_pos = ops.keltner_channel(df, 'close', 20, 2.0)
                        df['kc_lower'] = kc_lower
                elif tech == 'kc_position':
                    if 'kc_position' not in df.columns:
                        kc_upper, kc_lower, kc_pos = ops.keltner_channel(df, 'close', 20, 2.0)
                        df['kc_position'] = kc_pos

                # Ichimoku Cloud
                elif tech == 'ichimoku_conversion':
                    conv, base, span_a, span_b = ops.ichimoku_cloud(df, 9, 26, 52)
                    df['ichimoku_conversion'] = conv
                    df['ichimoku_base'] = base
                    df['ichimoku_span_a'] = span_a
                    df['ichimoku_span_b'] = span_b
                    print(f"[OK] ichimoku_conversion, base, span_a, span_b")
                    continue
                elif tech == 'ichimoku_base':
                    if 'ichimoku_base' not in df.columns:
                        conv, base, span_a, span_b = ops.ichimoku_cloud(df, 9, 26, 52)
                        df['ichimoku_base'] = base
                elif tech == 'ichimoku_span_a':
                    if 'ichimoku_span_a' not in df.columns:
                        conv, base, span_a, span_b = ops.ichimoku_cloud(df, 9, 26, 52)
                        df['ichimoku_span_a'] = span_a
                elif tech == 'ichimoku_span_b':
                    if 'ichimoku_span_b' not in df.columns:
                        conv, base, span_a, span_b = ops.ichimoku_cloud(df, 9, 26, 52)
                        df['ichimoku_span_b'] = span_b

                # Parabolic SAR
                elif tech == 'psar':
                    psar, psar_dir = ops.parabolic_sar(df, 0.02, 0.2)
                    df['psar'] = psar
                    df['psar_direction'] = psar_dir
                    print(f"[OK] psar and psar_direction")
                    continue
                elif tech == 'psar_direction':
                    if 'psar_direction' not in df.columns:
                        psar, psar_dir = ops.parabolic_sar(df, 0.02, 0.2)
                        df['psar_direction'] = psar_dir

                # SuperTrend
                elif tech == 'supertrend':
                    st, st_dir = ops.supertrend(df, 10, 3.0)
                    df['supertrend'] = st
                    df['supertrend_direction'] = st_dir
                    print(f"[OK] supertrend and supertrend_direction")
                    continue
                elif tech == 'supertrend_direction':
                    if 'supertrend_direction' not in df.columns:
                        st, st_dir = ops.supertrend(df, 10, 3.0)
                        df['supertrend_direction'] = st_dir

                # VWAP ratio
                elif tech == 'vwap_ratio':
                    df[tech] = ops.vwap_ratio(df, 20)

                # Statistical features
                elif tech == 'returns_std_20d':
                    df[tech] = ops.ts_stddev(df, 'returns_1d', 20) if 'returns_1d' in df.columns else ops.ts_stddev(df, 'close', 20)
                elif tech == 'returns_skew_20d':
                    df[tech] = ops.returns_skewness(df, 'close', 20)
                elif tech == 'returns_kurt_20d':
                    df[tech] = ops.returns_kurtosis(df, 'close', 20)

                # Momentum
                elif tech == 'momentum_10d':
                    df[tech] = ops.momentum(df, 'close', 10)
                elif tech == 'momentum_20d':
                    df[tech] = ops.momentum(df, 'close', 20)
                elif tech == 'momentum_50d':
                    df[tech] = ops.momentum(df, 'close', 50)

                # ROC (Rate of Change)
                elif tech == 'roc_10':
                    df[tech] = ops.roc(df, 'close', 10)
                elif tech == 'roc_20':
                    df[tech] = ops.roc(df, 'close', 20)

                # Directional features
                elif tech == 'price_above_sma_20':
                    df[tech] = ops.price_above_sma(df, 20)
                elif tech == 'price_above_ema_50':
                    df[tech] = ops.price_above_ema(df, 50)
                elif tech == 'golden_cross':
                    df[tech] = ops.golden_cross(df)
                elif tech == 'adx_strong_trend':
                    df[tech] = ops.adx_strong_trend(df, threshold=25.0)
                elif tech == 'macd_positive':
                    df[tech] = ops.macd_positive(df)
                elif tech == 'consecutive_up_days':
                    df[tech] = ops.consecutive_up_days(df, window=10)
                elif tech == 'consecutive_down_days':
                    df[tech] = ops.consecutive_down_days(df, window=10)
                elif tech == 'returns_1d_positive':
                    df[tech] = ops.returns_1d_positive(df)
                elif tech == 'volume_increasing':
                    df[tech] = ops.volume_increasing(df, window=5)
                elif tech == 'rsi_oversold':
                    df[tech] = ops.rsi_oversold(df, window=14, threshold=30.0)
                elif tech == 'bb_squeeze':
                    df[tech] = ops.bb_squeeze(df, window=20, threshold=0.01)

                else:
                    print(f"Unknown technical feature: {tech}")
                    continue

                print(f"[OK] {tech}")

            except Exception as e:
                print(f"[FAIL] {tech}: {e}")
                import traceback
                traceback.print_exc()

        return df

    def compute_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market environment features (SPY, VIX, QQQ, DIA, IWM, TLT, GLD, USO, UUP).
        EXPANDED: More comprehensive market context for maximum accuracy.
        """
        try:
            from . import operators as ops
        except ImportError:
            import operators as ops
        import yfinance as yf

        if 'market_features' not in self.config['features']:
            return df

        print("\n" + "=" * 60)
        print("Computing market features...")
        print("=" * 60)

        # Get unique dates from the dataframe
        start_date = df['date'].min() - pd.Timedelta(days=300)
        end_date = df['date'].max() + pd.Timedelta(days=1)

        # Dictionary to store all market data
        market_data = {}

        # Define market symbols and their features
        market_symbols = {
            'SPY': ['spy_returns_1d', 'spy_returns_5d', 'spy_returns_20d', 'spy_volatility_20d'],
            '^VIX': ['vix_level', 'vix_change_1d', 'vix_change_5d'],
            'QQQ': ['qqq_returns_1d', 'qqq_returns_5d'],
            'DIA': ['dia_returns_1d', 'dia_returns_5d'],
            'IWM': ['iwm_returns_1d', 'iwm_returns_5d'],
            'TLT': ['tlt_returns_1d', 'tlt_returns_5d'],
            'GLD': ['gld_returns_1d', 'gld_returns_5d'],
            'USO': ['uso_returns_1d', 'uso_returns_5d'],
            'UUP': ['uup_returns_1d', 'uup_returns_5d']
        }

        # Fetch each market symbol
        for symbol, feature_list in market_symbols.items():
            # Check if any feature from this symbol is requested
            if not any(feat in self.config['features']['market_features'] for feat in feature_list):
                continue

            try:
                print(f"\n  Fetching {symbol} data...")
                ticker = yf.Ticker(symbol)
                ticker_df = ticker.history(start=start_date, end=end_date)

                if len(ticker_df) == 0:
                    print(f"  [WARN] No data for {symbol}")
                    continue

                ticker_df = ticker_df.reset_index()
                ticker_df.columns = [c.lower() for c in ticker_df.columns]
                ticker_df['symbol'] = symbol

                # Compute features for this symbol
                features_dict = {'date': ticker_df['date']}

                # SPY features
                if symbol == 'SPY':
                    features_dict['spy_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['spy_returns_5d'] = ops.returns(ticker_df, 'close', 5)
                    features_dict['spy_returns_20d'] = ops.returns(ticker_df, 'close', 20)
                    features_dict['spy_volatility_20d'] = ops.volatility(ticker_df, 20, 'close')

                # VIX features
                elif symbol == '^VIX':
                    features_dict['vix_level'] = ticker_df['close']
                    features_dict['vix_change_1d'] = ops.delta(ticker_df, 'close', 1)
                    features_dict['vix_change_5d'] = ops.delta(ticker_df, 'close', 5)

                # QQQ features
                elif symbol == 'QQQ':
                    features_dict['qqq_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['qqq_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # DIA features
                elif symbol == 'DIA':
                    features_dict['dia_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['dia_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # IWM features
                elif symbol == 'IWM':
                    features_dict['iwm_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['iwm_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # TLT features
                elif symbol == 'TLT':
                    features_dict['tlt_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['tlt_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # GLD features
                elif symbol == 'GLD':
                    features_dict['gld_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['gld_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # USO features
                elif symbol == 'USO':
                    features_dict['uso_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['uso_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                # UUP features
                elif symbol == 'UUP':
                    features_dict['uup_returns_1d'] = ops.returns(ticker_df, 'close', 1)
                    features_dict['uup_returns_5d'] = ops.returns(ticker_df, 'close', 5)

                market_data[symbol] = pd.DataFrame(features_dict)
                print(f"  [OK] {symbol} - {len(ticker_df)} days")

            except Exception as e:
                print(f"  [FAIL] {symbol}: {e}")
                continue

        # Compute market breadth features (from stock universe)
        if 'market_breadth_1d' in self.config['features']['market_features'] or \
           'market_breadth_5d' in self.config['features']['market_features']:
            try:
                print(f"\n  Computing market breadth...")

                # Calculate % of stocks up each day
                breadth_1d = df.groupby('date').apply(
                    lambda g: (g['close'] > g.groupby('symbol')['close'].shift(1)).mean()
                ).reset_index()
                breadth_1d.columns = ['date', 'market_breadth_1d']

                breadth_5d = df.groupby('date').apply(
                    lambda g: (g['close'] > g.groupby('symbol')['close'].shift(5)).mean()
                ).reset_index()
                breadth_5d.columns = ['date', 'market_breadth_5d']

                market_data['breadth'] = breadth_1d.merge(breadth_5d, on='date', how='outer')
                print(f"  [OK] market_breadth")

            except Exception as e:
                print(f"  [FAIL] market_breadth: {e}")

        # Merge all market features into main dataframe
        print(f"\n  Merging market features...")
        for feat in self.config['features']['market_features']:
            try:
                # Find which market dataset contains this feature
                for symbol, data_df in market_data.items():
                    if feat in data_df.columns:
                        df = df.merge(data_df[['date', feat]], on='date', how='left')
                        print(f"  [OK] {feat}")
                        break
                else:
                    print(f"  [WARN] {feat} not computed")

            except Exception as e:
                print(f"  [FAIL] {feat}: {e}")

        return df

    def _compute_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        from . import operators as ops

        delta = ops.delta(df, 'close', 1)

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use exponential moving average
        df_temp = df.copy()
        df_temp['gain'] = gain
        df_temp['loss'] = loss

        avg_gain = df_temp.groupby('symbol')['gain'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
        avg_loss = df_temp.groupby('symbol')['loss'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """Compute MACD (Moving Average Convergence Divergence)."""
        ema_fast = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=fast, adjust=False).mean()
        )
        ema_slow = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=slow, adjust=False).mean()
        )

        macd = ema_fast - ema_slow

        return macd

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate prediction labels.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with 'label' column added
        """
        print(f"\n{'=' * 60}")
        print("Generating labels...")
        print(f"{'=' * 60}\n")

        horizon = self.label_config['horizon']
        label_type = self.label_config['type']
        cost_buffer = self.label_config.get('cost_buffer', 0.002)

        print(f"Label type: {label_type}")
        print(f"Horizon: {horizon} days")
        print(f"Cost buffer: {cost_buffer * 100:.2f}%")

        # Calculate future returns
        df['future_return'] = df.groupby('symbol')['close'].transform(
            lambda x: np.log(x.shift(-horizon) / x)
        )

        if label_type == 'classification':
            # Binary classification: 1 if return > cost_buffer, else 0
            df['label'] = (df['future_return'] > cost_buffer).astype(int)

            print(f"\nLabel distribution:")
            print(df['label'].value_counts(dropna=False))
            print(f"Positive rate: {df['label'].mean() * 100:.2f}%")

        elif label_type == 'regression':
            # Regression: predict future return directly
            df['label'] = df['future_return']

            print(f"\nLabel statistics:")
            print(df['label'].describe())

        else:
            raise ValueError(f"Unknown label type: {label_type}")

        # Drop the last `horizon` days per symbol (no labels available)
        df_labeled = df.groupby('symbol').apply(
            lambda g: g.iloc[:-horizon] if len(g) > horizon else g.iloc[:0]
        ).reset_index(drop=True)

        rows_dropped = len(df) - len(df_labeled)
        print(f"\nDropped {rows_dropped} rows without labels (last {horizon} days per symbol)")

        return df_labeled

    def clean_dataset(self, df: pd.DataFrame, feature_cols: List[str]) -> tuple:
        """
        Clean dataset: handle NaN, inf, etc.

        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names

        Returns:
            Tuple of (cleaned DataFrame, updated feature_cols list)
        """
        print(f"\n{'=' * 60}")
        print("Cleaning dataset...")
        print(f"{'=' * 60}\n")

        initial_rows = len(df)

        # Replace inf with NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Check NaN percentage
        nan_pct = df[feature_cols].isna().mean() * 100
        high_nan_features = nan_pct[nan_pct > 50].index.tolist()

        if high_nan_features:
            print(f"Warning: {len(high_nan_features)} features have >50% NaN:")
            for feat in high_nan_features[:10]:  # Show first 10
                print(f"  - {feat}: {nan_pct[feat]:.1f}%")

        # Drop features with >90% NaN (likely failed to compute)
        very_bad_features = nan_pct[nan_pct > 90].index.tolist()
        if very_bad_features:
            print(f"\nDropping {len(very_bad_features)} features with >90% NaN:")
            for feat in very_bad_features:
                print(f"  - {feat}: {nan_pct[feat]:.1f}%")
            feature_cols = [f for f in feature_cols if f not in very_bad_features]
            print(f"Remaining features: {len(feature_cols)}")

        # Drop rows where label is NaN
        df = df[df['label'].notna()]

        # Drop rows where ALL features are NaN
        df = df.dropna(subset=feature_cols, how='all')

        # Forward-fill remaining NaNs (within each symbol)
        print("\nForward-filling NaN values per symbol...")
        df[feature_cols] = df.groupby('symbol')[feature_cols].ffill()

        # Backward-fill any remaining NaNs
        df[feature_cols] = df.groupby('symbol')[feature_cols].bfill()

        # Fill any remaining NaNs with 0 (should be rare)
        remaining_nans = df[feature_cols].isna().sum().sum()
        if remaining_nans > 0:
            print(f"\nFilling {remaining_nans} remaining NaN values with 0...")
            df[feature_cols] = df[feature_cols].fillna(0)

        # Final check - should have no NaNs now
        final_nans = df[feature_cols + ['label']].isna().sum().sum()
        if final_nans > 0:
            print(f"Warning: Still have {final_nans} NaN values after cleaning")
            # Drop these rows as last resort
            df = df.dropna(subset=feature_cols + ['label'])

        rows_dropped = initial_rows - len(df)
        print(f"\nDropped {rows_dropped} rows ({rows_dropped / initial_rows * 100:.2f}%) due to missing labels")
        print(f"Final dataset: {len(df):,} rows")
        print(f"Final features: {len(feature_cols)}")

        return df, feature_cols

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_start_date: str = None
    ) -> tuple:
        """
        Split data into train and test sets.

        Args:
            df: Full dataset
            test_start_date: Date to start test set (default: last 20% of data)

        Returns:
            (df_train, df_test)
        """
        if test_start_date is None:
            # Use last 20% for test
            dates = df['date'].unique()
            test_idx = int(len(dates) * 0.8)
            test_start_date = dates[test_idx]

        df_train = df[df['date'] < test_start_date].copy()
        df_test = df[df['date'] >= test_start_date].copy()

        print(f"\n{'=' * 60}")
        print("Train/Test Split")
        print(f"{'=' * 60}")
        print(f"Train: {df_train['date'].min()} to {df_train['date'].max()} ({len(df_train):,} rows)")
        print(f"Test:  {df_test['date'].min()} to {df_test['date'].max()} ({len(df_test):,} rows)")

        return df_train, df_test

    def export_manifest(
        self,
        feature_cols: List[str],
        output_path: str = None
    ):
        """
        Export feature manifest for Go inference.

        Args:
            feature_cols: List of feature column names
            output_path: Path to save manifest JSON
        """
        if output_path is None:
            output_path = Path(__file__).parent.parent / self.config['model']['manifest_output']

        manifest = {
            'feature_names': feature_cols,
            'num_features': len(feature_cols),
            'lookback_days': self.max_lookback,
            'label_type': self.label_config['type'],
            'label_horizon': self.label_config['horizon'],
            'missing_policy': 'forward_fill',
            'version': '1.0',
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\nFeature manifest saved to {output_path}")

    def build_full_pipeline(self) -> Dict:
        """
        Run full dataset building pipeline.

        Returns:
            Dict with dataset, manifest, and metadata
        """
        # 1. Load data
        df = self.load_data()

        # 2. Compute features
        df_features = self.compute_features(df)

        # 2.5. Compute market features (Direction 4)
        df_features = self.compute_market_features(df_features)

        # 3. Generate labels
        df_labeled = self.generate_labels(df_features)

        # 4. Get feature columns (exclude metadata + label)
        meta_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'label']
        feature_cols = [c for c in df_labeled.columns if c not in meta_cols]

        print(f"\n{'=' * 60}")
        print(f"Feature columns ({len(feature_cols)}):")
        print(f"{'=' * 60}")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:3d}. {col}")

        # 5. Clean dataset (returns cleaned df and updated feature_cols)
        df_clean, feature_cols = self.clean_dataset(df_labeled, feature_cols)

        # 5.5. Save full dataset with all features to configured parquet_path
        parquet_path = Path(__file__).parent.parent / self.config['data']['parquet_path']
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(parquet_path, index=False)
        print(f"\nSaved full dataset to: {parquet_path}")
        print(f"  Shape: {df_clean.shape}")
        print(f"  Columns: {list(df_clean.columns[:5])}... + {len(df_clean.columns) - 5} more")

        # 6. Split train/test
        df_train, df_test = self.split_train_test(df_clean)

        # 7. Export manifest (with updated feature list)
        self.export_manifest(feature_cols)

        # 8. Save datasets
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_path = artifacts_dir / "train_dataset.parquet"
        test_path = artifacts_dir / "test_dataset.parquet"

        df_train.to_parquet(train_path, index=False)
        df_test.to_parquet(test_path, index=False)

        print(f"\nSaved datasets:")
        print(f"  - Train: {train_path}")
        print(f"  - Test:  {test_path}")

        return {
            'df_train': df_train,
            'df_test': df_test,
            'feature_cols': feature_cols,
            'label_col': 'label',
        }


def main():
    """Command-line entry point."""
    print("=" * 60)
    print("Dataset Builder")
    print("=" * 60)

    builder = DatasetBuilder()
    result = builder.build_full_pipeline()

    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)

    print(f"\nTrain shape: {result['df_train'].shape}")
    print(f"Test shape: {result['df_test'].shape}")
    print(f"Features: {len(result['feature_cols'])}")

    print("\nSample training data:")
    print(result['df_train'][['date', 'symbol', 'close', 'label'] + result['feature_cols'][:3]].head(10))


if __name__ == "__main__":
    main()
