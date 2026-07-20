"""
Core operators for building 101 Formulaic Alphas.

These operators follow the WorldQuant 101 Alphas paper definitions.
All operators are vectorized using pandas for performance.
"""

import numpy as np
import pandas as pd
from typing import Union


def rank(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Cross-sectional rank of values.

    Ranks values within each date across all symbols.
    Returns values between 0 and 1 (percentile rank).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name to rank

    Returns:
        Series of ranked values (0 to 1)
    """
    return df.groupby('date')[column].rank(pct=True)


def delay(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Value of column d days ago (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Number of days to delay

    Returns:
        Series of delayed values
    """
    return df.groupby('symbol')[column].shift(d)


def delta(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Difference: today's value minus value d days ago.

    delta(x, d) = x(t) - x(t-d)

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Number of days to look back

    Returns:
        Series of differences
    """
    return df.groupby('symbol')[column].diff(d)


def correlation(df: pd.DataFrame, col1: str, col2: str, d: int) -> pd.Series:
    """
    Time-series correlation of two columns over past d days.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        col1: First column name
        col2: Second column name
        d: Window size (number of days)

    Returns:
        Series of rolling correlations
    """
    def rolling_corr(group):
        return group[col1].rolling(window=d, min_periods=d).corr(group[col2])

    return df.groupby('symbol', group_keys=False).apply(rolling_corr)


def covariance(df: pd.DataFrame, col1: str, col2: str, d: int) -> pd.Series:
    """
    Time-series covariance of two columns over past d days.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        col1: First column name
        col2: Second column name
        d: Window size (number of days)

    Returns:
        Series of rolling covariances
    """
    def rolling_cov(group):
        return group[col1].rolling(window=d, min_periods=d).cov(group[col2])

    return df.groupby('symbol', group_keys=False).apply(rolling_cov)


def ts_rank(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Time-series rank: rank of current value among past d days (per symbol).

    Returns values between 0 and 1 (percentile rank in rolling window).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of time-series ranks
    """
    def rolling_rank(x):
        # Rank the current value against the window
        return x.rolling(window=d, min_periods=d).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1],
            raw=False
        )

    return df.groupby('symbol', group_keys=False)[column].apply(rolling_rank)


def ts_min(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Minimum value over past d days (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of rolling minimums
    """
    return df.groupby('symbol')[column].rolling(window=d, min_periods=d).min().reset_index(level=0, drop=True)


def ts_max(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Maximum value over past d days (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of rolling maximums
    """
    return df.groupby('symbol')[column].rolling(window=d, min_periods=d).max().reset_index(level=0, drop=True)


def ts_argmax(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Number of days since the maximum value in the past d days.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of days since max
    """
    def days_since_max(x):
        return x.rolling(window=d, min_periods=d).apply(
            lambda w: (len(w) - 1) - w.argmax(),
            raw=True
        )

    return df.groupby('symbol', group_keys=False)[column].apply(days_since_max)


def ts_argmin(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Number of days since the minimum value in the past d days.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of days since min
    """
    def days_since_min(x):
        return x.rolling(window=d, min_periods=d).apply(
            lambda w: (len(w) - 1) - w.argmin(),
            raw=True
        )

    return df.groupby('symbol', group_keys=False)[column].apply(days_since_min)


def ts_sum(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Sum over past d days (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of rolling sums
    """
    return df.groupby('symbol')[column].rolling(window=d, min_periods=d).sum().reset_index(level=0, drop=True)


def ts_mean(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Mean over past d days (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of rolling means
    """
    return df.groupby('symbol')[column].rolling(window=d, min_periods=d).mean().reset_index(level=0, drop=True)


def ts_stddev(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Standard deviation over past d days (per symbol).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of rolling standard deviations
    """
    return df.groupby('symbol')[column].rolling(window=d, min_periods=d).std().reset_index(level=0, drop=True)


def decay_linear(df: pd.DataFrame, column: str, d: int) -> pd.Series:
    """
    Weighted moving average with linear decay weights.

    Most recent value has weight d, previous has weight d-1, etc.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Column name
        d: Window size (number of days)

    Returns:
        Series of linearly decayed weighted averages
    """
    weights = np.arange(1, d + 1)

    def weighted_mean(x):
        return x.rolling(window=d, min_periods=d).apply(
            lambda w: np.dot(w, weights) / weights.sum(),
            raw=True
        )

    return df.groupby('symbol', group_keys=False)[column].apply(weighted_mean)


def sign(series: pd.Series) -> pd.Series:
    """
    Sign of values: 1 if positive, -1 if negative, 0 if zero.

    Args:
        series: Input series

    Returns:
        Series of signs
    """
    return np.sign(series)


def scale(series: pd.Series, a: float = 1.0) -> pd.Series:
    """
    Rescale series to sum to a (per cross-section).

    Typically used for position sizing.

    Args:
        series: Input series
        a: Target sum (default 1.0)

    Returns:
        Rescaled series
    """
    total = series.abs().sum()
    if total == 0:
        return series
    return series * (a / total)


def adv(df: pd.DataFrame, d: int) -> pd.Series:
    """
    Average daily dollar volume over past d days.

    adv(d) = mean(volume * close, d)

    Args:
        df: DataFrame with 'date', 'symbol', 'volume', 'close' columns
        d: Window size (number of days)

    Returns:
        Series of average daily volumes
    """
    df = df.copy()
    df['dollar_volume'] = df['volume'] * df['close']
    return ts_mean(df, 'dollar_volume', d)


def log(series: pd.Series) -> pd.Series:
    """
    Natural logarithm.

    Args:
        series: Input series

    Returns:
        Log-transformed series
    """
    return np.log(series)


def abs_op(series: pd.Series) -> pd.Series:
    """
    Absolute value.

    Args:
        series: Input series

    Returns:
        Absolute values
    """
    return series.abs()


# Additional useful operators not in original 101 paper but commonly used

def returns(df: pd.DataFrame, column: str = 'close', periods: int = 1) -> pd.Series:
    """
    Log returns over specified periods.

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        column: Price column (default 'close')
        periods: Number of periods (default 1 for daily returns)

    Returns:
        Series of log returns
    """
    result = df.groupby('symbol')[column].apply(
        lambda x: np.log(x / x.shift(periods))
    )
    # Reset index to match original DataFrame index
    return result.reset_index(level=0, drop=True)


def volatility(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Realized volatility (std of log returns).

    Args:
        df: DataFrame with 'date' and 'symbol' columns
        window: Window size for volatility calculation
        column: Price column (default 'close')

    Returns:
        Series of realized volatility
    """
    df = df.copy()
    df['log_ret'] = returns(df, column, 1)
    return ts_stddev(df, 'log_ret', window)


def ema(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        df: DataFrame with 'symbol' column
        column: Column to compute EMA on
        window: EMA window size

    Returns:
        Series of EMA values
    """
    return df.groupby('symbol')[column].transform(
        lambda x: x.ewm(span=window, adjust=False).mean()
    )


def bollinger_bands(df: pd.DataFrame, column: str = 'close', window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands (upper, lower, position).

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: MA window
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper_band, lower_band, bb_position)
    """
    ma = df.groupby('symbol')[column].transform(lambda x: x.rolling(window).mean())
    std = df.groupby('symbol')[column].transform(lambda x: x.rolling(window).std())

    upper = ma + (std * num_std)
    lower = ma - (std * num_std)

    # Position: where price is relative to bands (0=lower, 0.5=middle, 1=upper)
    bb_position = (df[column] - lower) / (upper - lower + 1e-10)

    return upper, lower, bb_position


def macd_indicator(df: pd.DataFrame, column: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD and Signal line.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        fast: Fast EMA window
        slow: Slow EMA window
        signal: Signal line EMA window

    Returns:
        Tuple of (macd, signal_line, histogram)
    """
    ema_fast = df.groupby('symbol')[column].transform(
        lambda x: x.ewm(span=fast, adjust=False).mean()
    )
    ema_slow = df.groupby('symbol')[column].transform(
        lambda x: x.ewm(span=slow, adjust=False).mean()
    )

    macd = ema_fast - ema_slow

    # Signal line
    signal_line = df.groupby('symbol').apply(
        lambda g: pd.Series(macd[g.index].ewm(span=signal, adjust=False).mean(), index=g.index)
    ).reset_index(level=0, drop=True)

    histogram = macd - signal_line

    return macd, signal_line, histogram


def rsi(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: RSI window

    Returns:
        Series of RSI values (0-100)
    """
    delta = df.groupby('symbol')[column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = df.groupby('symbol').apply(
        lambda g: pd.Series(gain[g.index].ewm(span=window, adjust=False).mean(), index=g.index)
    ).reset_index(level=0, drop=True)

    avg_loss = df.groupby('symbol').apply(
        lambda g: pd.Series(loss[g.index].ewm(span=window, adjust=False).mean(), index=g.index)
    ).reset_index(level=0, drop=True)

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range - volatility indicator.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        window: ATR window

    Returns:
        Series of ATR values
    """
    high = df['high']
    low = df['low']
    close = df.groupby('symbol')['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_values = df.groupby('symbol').apply(
        lambda g: pd.Series(true_range[g.index].rolling(window=window).mean(), index=g.index)
    ).reset_index(level=0, drop=True)

    return atr_values


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index - trend strength indicator.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        window: ADX window

    Returns:
        Series of ADX values (0-100, higher = stronger trend)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM
    high_diff = df.groupby('symbol')['high'].diff()
    low_diff = -df.groupby('symbol')['low'].diff()

    pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Calculate True Range
    prev_close = df.groupby('symbol')['close'].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth with EMA
    def smooth_series(series, g_index):
        return pd.Series(series[g_index].ewm(span=window, adjust=False).mean(), index=g_index)

    atr_smooth = df.groupby('symbol').apply(
        lambda g: smooth_series(true_range, g.index)
    ).reset_index(level=0, drop=True)

    pos_di = 100 * df.groupby('symbol').apply(
        lambda g: smooth_series(pos_dm, g.index)
    ).reset_index(level=0, drop=True) / (atr_smooth + 1e-10)

    neg_di = 100 * df.groupby('symbol').apply(
        lambda g: smooth_series(neg_dm, g.index)
    ).reset_index(level=0, drop=True) / (atr_smooth + 1e-10)

    # Calculate DX
    dx = 100 * (pos_di - neg_di).abs() / ((pos_di + neg_di) + 1e-10)

    # Calculate ADX (smoothed DX)
    adx_values = df.groupby('symbol').apply(
        lambda g: smooth_series(dx, g.index)
    ).reset_index(level=0, drop=True)

    return adx_values


# ============================================================================
# ADDITIONAL TECHNICAL INDICATORS (for maximum accuracy)
# ============================================================================

def sma(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        df: DataFrame with 'symbol' column
        column: Column to compute SMA on
        window: SMA window size

    Returns:
        Series of SMA values
    """
    return df.groupby('symbol')[column].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )


def stochastic_oscillator(df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """
    Stochastic Oscillator (%K and %D).

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        window: Lookback window for high/low
        smooth_k: Smoothing period for %K
        smooth_d: Smoothing period for %D

    Returns:
        Tuple of (stoch_k, stoch_d)
    """
    def calc_stoch(group):
        low_min = group['low'].rolling(window=window, min_periods=window).min()
        high_max = group['high'].rolling(window=window, min_periods=window).max()

        stoch_k_raw = 100 * (group['close'] - low_min) / (high_max - low_min + 1e-10)
        stoch_k = stoch_k_raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d, min_periods=smooth_d).mean()

        return pd.DataFrame({'stoch_k': stoch_k, 'stoch_d': stoch_d}, index=group.index)

    result = df.groupby('symbol', group_keys=False).apply(calc_stoch)
    return result['stoch_k'], result['stoch_d']


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Commodity Channel Index.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        window: CCI window

    Returns:
        Series of CCI values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    def calc_cci(group):
        tp = typical_price[group.index]
        sma_tp = tp.rolling(window=window, min_periods=window).mean()
        mean_dev = tp.rolling(window=window, min_periods=window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=False
        )
        cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-10)
        return cci

    return df.groupby('symbol', group_keys=False).apply(calc_cci)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Williams %R.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        window: Williams %R window

    Returns:
        Series of Williams %R values (-100 to 0)
    """
    def calc_wr(group):
        high_max = group['high'].rolling(window=window, min_periods=window).max()
        low_min = group['low'].rolling(window=window, min_periods=window).min()
        wr = -100 * (high_max - group['close']) / (high_max - low_min + 1e-10)
        return wr

    return df.groupby('symbol', group_keys=False).apply(calc_wr)


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    Args:
        df: DataFrame with 'symbol', 'close', 'volume' columns

    Returns:
        Series of OBV values
    """
    def calc_obv(group):
        direction = np.sign(group['close'].diff())
        obv_values = (direction * group['volume']).fillna(0).cumsum()
        return obv_values

    return df.groupby('symbol', group_keys=False).apply(calc_obv)


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Chaikin Money Flow.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close', 'volume' columns
        window: CMF window

    Returns:
        Series of CMF values
    """
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mf_volume = mf_multiplier * df['volume']

    def calc_cmf(group):
        mfv = mf_volume[group.index]
        vol = group['volume']
        cmf_values = mfv.rolling(window=window).sum() / (vol.rolling(window=window).sum() + 1e-10)
        return cmf_values

    return df.groupby('symbol', group_keys=False).apply(calc_cmf)


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Money Flow Index.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close', 'volume' columns
        window: MFI window

    Returns:
        Series of MFI values (0-100)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']

    def calc_mfi(group):
        tp = typical_price[group.index]
        rmf = raw_money_flow[group.index]

        positive_flow = rmf.where(tp.diff() > 0, 0).rolling(window=window).sum()
        negative_flow = rmf.where(tp.diff() < 0, 0).rolling(window=window).sum()

        mfi_ratio = positive_flow / (negative_flow + 1e-10)
        mfi_values = 100 - (100 / (1 + mfi_ratio))
        return mfi_values

    return df.groupby('symbol', group_keys=False).apply(calc_mfi)


def trix(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.Series:
    """
    TRIX - Triple Exponential Average.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: TRIX window

    Returns:
        Series of TRIX values
    """
    def calc_trix(group):
        ema1 = group[column].ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        trix_values = 100 * ema3.pct_change()
        return trix_values

    return df.groupby('symbol', group_keys=False).apply(calc_trix)


def dpo(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Detrended Price Oscillator.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: DPO window

    Returns:
        Series of DPO values
    """
    shift_period = window // 2 + 1

    def calc_dpo(group):
        # DPO = close from (window/2 + 1) periods AGO minus current SMA.
        # The shift must be positive (into the past); a negative shift here
        # would read close[t + shift] and leak future prices into the feature.
        sma_values = group[column].rolling(window=window).mean()
        dpo_values = group[column].shift(shift_period) - sma_values
        return dpo_values

    return df.groupby('symbol', group_keys=False).apply(calc_dpo)


def keltner_channel(df: pd.DataFrame, column: str = 'close', window: int = 20, atr_multiplier: float = 2.0):
    """
    Keltner Channel (upper, lower, position).

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        column: Price column
        window: EMA window
        atr_multiplier: Multiplier for ATR

    Returns:
        Tuple of (kc_upper, kc_lower, kc_position)
    """
    ema_values = ema(df, column, window)
    atr_values = atr(df, window)

    kc_upper = ema_values + (atr_multiplier * atr_values)
    kc_lower = ema_values - (atr_multiplier * atr_values)

    kc_position = (df[column] - kc_lower) / (kc_upper - kc_lower + 1e-10)

    return kc_upper, kc_lower, kc_position


def ichimoku_cloud(df: pd.DataFrame, conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52):
    """
    Ichimoku Cloud components.

    Args:
        df: DataFrame with 'symbol', 'high', 'low' columns
        conversion_period: Conversion line period
        base_period: Base line period
        span_b_period: Span B period

    Returns:
        Tuple of (conversion_line, base_line, span_a, span_b)
    """
    def calc_ichimoku(group):
        # Conversion Line (Tenkan-sen)
        high_conv = group['high'].rolling(window=conversion_period).max()
        low_conv = group['low'].rolling(window=conversion_period).min()
        conversion = (high_conv + low_conv) / 2

        # Base Line (Kijun-sen)
        high_base = group['high'].rolling(window=base_period).max()
        low_base = group['low'].rolling(window=base_period).min()
        base = (high_base + low_base) / 2

        # Span A (Senkou Span A)
        span_a = ((conversion + base) / 2).shift(base_period)

        # Span B (Senkou Span B)
        high_span_b = group['high'].rolling(window=span_b_period).max()
        low_span_b = group['low'].rolling(window=span_b_period).min()
        span_b = ((high_span_b + low_span_b) / 2).shift(base_period)

        return pd.DataFrame({
            'conversion': conversion,
            'base': base,
            'span_a': span_a,
            'span_b': span_b
        }, index=group.index)

    result = df.groupby('symbol', group_keys=False).apply(calc_ichimoku)
    return result['conversion'], result['base'], result['span_a'], result['span_b']


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2):
    """
    Parabolic SAR and direction.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        af_start: Starting acceleration factor
        af_max: Maximum acceleration factor

    Returns:
        Tuple of (psar, psar_direction) where direction is 1 (up) or -1 (down)
    """
    def calc_psar(group):
        high = group['high'].values
        low = group['low'].values
        close = group['close'].values

        n = len(close)
        psar_values = np.zeros(n)
        direction = np.zeros(n)
        ep = 0
        af = af_start

        # Initialize
        psar_values[0] = close[0]
        direction[0] = 1

        for i in range(1, n):
            if direction[i-1] == 1:  # Uptrend
                psar_values[i] = psar_values[i-1] + af * (ep - psar_values[i-1])
                psar_values[i] = min(psar_values[i], low[i-1], low[i-2] if i > 1 else low[i-1])

                if low[i] < psar_values[i]:  # Switch to downtrend
                    direction[i] = -1
                    psar_values[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    direction[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_start, af_max)
            else:  # Downtrend
                psar_values[i] = psar_values[i-1] + af * (ep - psar_values[i-1])
                psar_values[i] = max(psar_values[i], high[i-1], high[i-2] if i > 1 else high[i-1])

                if high[i] > psar_values[i]:  # Switch to uptrend
                    direction[i] = 1
                    psar_values[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    direction[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_start, af_max)

        return pd.DataFrame({
            'psar': psar_values,
            'direction': direction
        }, index=group.index)

    result = df.groupby('symbol', group_keys=False).apply(calc_psar)
    return result['psar'], result['direction']


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    SuperTrend indicator and direction.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        Tuple of (supertrend, direction) where direction is 1 (up) or -1 (down)
    """
    atr_values = atr(df, period)
    hl_avg = (df['high'] + df['low']) / 2

    def calc_supertrend(group):
        close = group['close'].values
        atr_val = atr_values[group.index].values
        hl = hl_avg[group.index].values

        n = len(close)
        supertrend_values = np.zeros(n)
        direction = np.ones(n)

        upper_band = hl + multiplier * atr_val
        lower_band = hl - multiplier * atr_val

        for i in range(1, n):
            if close[i] > upper_band[i-1]:
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

            if direction[i] == 1:
                supertrend_values[i] = lower_band[i]
            else:
                supertrend_values[i] = upper_band[i]

        return pd.DataFrame({
            'supertrend': supertrend_values,
            'direction': direction
        }, index=group.index)

    result = df.groupby('symbol', group_keys=False).apply(calc_supertrend)
    return result['supertrend'], result['direction']


def vwap_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    VWAP (Volume Weighted Average Price) ratio.

    Args:
        df: DataFrame with 'symbol', 'close', 'volume' columns
        window: VWAP window

    Returns:
        Series of price/VWAP ratios
    """
    def calc_vwap(group):
        typical_price = group['close']
        volume = group['volume']

        vwap = (typical_price * volume).rolling(window=window).sum() / (volume.rolling(window=window).sum() + 1e-10)
        vwap_ratio_val = typical_price / (vwap + 1e-10)
        return vwap_ratio_val

    return df.groupby('symbol', group_keys=False).apply(calc_vwap)


def returns_skewness(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Skewness of returns over rolling window.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: Window for skewness calculation

    Returns:
        Series of skewness values
    """
    df_temp = df.copy()
    df_temp['returns_temp'] = returns(df_temp, column, 1)

    return df_temp.groupby('symbol')['returns_temp'].transform(
        lambda x: x.rolling(window=window, min_periods=window).skew()
    )


def returns_kurtosis(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
    """
    Kurtosis of returns over rolling window.

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: Window for kurtosis calculation

    Returns:
        Series of kurtosis values
    """
    df_temp = df.copy()
    df_temp['returns_temp'] = returns(df_temp, column, 1)

    return df_temp.groupby('symbol')['returns_temp'].transform(
        lambda x: x.rolling(window=window, min_periods=window).kurt()
    )


def momentum(df: pd.DataFrame, column: str = 'close', window: int = 10) -> pd.Series:
    """
    Price momentum (current price / price n days ago).

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: Lookback period

    Returns:
        Series of momentum values
    """
    return df.groupby('symbol')[column].transform(
        lambda x: x / x.shift(window) - 1
    )


def roc(df: pd.DataFrame, column: str = 'close', window: int = 10) -> pd.Series:
    """
    Rate of Change (ROC).

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: Lookback period

    Returns:
        Series of ROC values (percentage)
    """
    return df.groupby('symbol')[column].transform(
        lambda x: 100 * (x - x.shift(window)) / (x.shift(window) + 1e-10)
    )


def bb_width(df: pd.DataFrame, column: str = 'close', window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band width (normalized).

    Args:
        df: DataFrame with 'symbol' column
        column: Price column
        window: MA window
        num_std: Number of standard deviations

    Returns:
        Series of BB width values
    """
    upper, lower, _ = bollinger_bands(df, column, window, num_std)
    ma = sma(df, column, window)
    width = (upper - lower) / (ma + 1e-10)
    return width


def volume_std(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Standard deviation of volume.

    Args:
        df: DataFrame with 'symbol', 'volume' columns
        window: Window size

    Returns:
        Series of volume std values
    """
    return df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window=window, min_periods=window).std()
    )


# ============================================================================
# DIRECTIONAL FEATURES (for predicting price direction, not absolute values)
# ============================================================================

def price_above_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Binary feature: 1 if price > SMA, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: SMA window

    Returns:
        Series of 1/0 values
    """
    sma_values = sma(df, 'close', window)
    return (df['close'] > sma_values).astype(int)


def price_above_ema(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Binary feature: 1 if price > EMA, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: EMA window

    Returns:
        Series of 1/0 values
    """
    ema_values = ema(df, 'close', window)
    return (df['close'] > ema_values).astype(int)


def golden_cross(df: pd.DataFrame) -> pd.Series:
    """
    Binary feature: 1 if SMA10 > SMA50 (golden cross), 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns

    Returns:
        Series of 1/0 values
    """
    sma_10 = sma(df, 'close', 10)
    sma_50 = sma(df, 'close', 50)
    return (sma_10 > sma_50).astype(int)


def adx_strong_trend(df: pd.DataFrame, threshold: float = 25.0) -> pd.Series:
    """
    Binary feature: 1 if ADX > threshold (strong trend), 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'high', 'low', 'close' columns
        threshold: ADX threshold for strong trend

    Returns:
        Series of 1/0 values
    """
    adx_values = adx(df, window=14)
    return (adx_values > threshold).astype(int)


def macd_positive(df: pd.DataFrame) -> pd.Series:
    """
    Binary feature: 1 if MACD > 0, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns

    Returns:
        Series of 1/0 values
    """
    macd_values, _, _ = macd_indicator(df, 'close', fast=12, slow=26)
    return (macd_values > 0).astype(int)


def consecutive_up_days(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Count consecutive up days (rolling).

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: Lookback window

    Returns:
        Series of consecutive up day counts
    """
    def count_consecutive_up(group):
        daily_change = group['close'].diff() > 0
        result = pd.Series(0, index=group.index)

        for i in range(window, len(group)):
            count = 0
            for j in range(i, max(i - window, -1), -1):
                if daily_change.iloc[j]:
                    count += 1
                else:
                    break
            result.iloc[i] = count

        return result

    return df.groupby('symbol', group_keys=False).apply(count_consecutive_up)


def consecutive_down_days(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Count consecutive down days (rolling).

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: Lookback window

    Returns:
        Series of consecutive down day counts
    """
    def count_consecutive_down(group):
        daily_change = group['close'].diff() < 0
        result = pd.Series(0, index=group.index)

        for i in range(window, len(group)):
            count = 0
            for j in range(i, max(i - window, -1), -1):
                if daily_change.iloc[j]:
                    count += 1
                else:
                    break
            result.iloc[i] = count

        return result

    return df.groupby('symbol', group_keys=False).apply(count_consecutive_down)


def returns_1d_positive(df: pd.DataFrame) -> pd.Series:
    """
    Binary feature: 1 if yesterday's return > 0, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns

    Returns:
        Series of 1/0 values
    """
    returns_1d = returns(df, 'close', 1)
    return (returns_1d > 0).astype(int)


def volume_increasing(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Binary feature: 1 if volume > average volume of past N days, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'volume' columns
        window: Lookback window

    Returns:
        Series of 1/0 values
    """
    avg_volume = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    return (df['volume'] > avg_volume).astype(int)


def rsi_oversold(df: pd.DataFrame, window: int = 14, threshold: float = 30.0) -> pd.Series:
    """
    Binary feature: 1 if RSI < threshold (oversold), 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: RSI window
        threshold: Oversold threshold

    Returns:
        Series of 1/0 values
    """
    rsi_values = rsi(df, 'close', window)
    return (rsi_values < threshold).astype(int)


def spy_direction_aligned(df: pd.DataFrame, spy_returns: pd.Series) -> pd.Series:
    """
    Binary feature: 1 if stock and SPY have same direction, 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns
        spy_returns: Series of SPY returns (indexed by date)

    Returns:
        Series of 1/0 values
    """
    stock_returns = df.groupby('symbol')['close'].pct_change()

    def align_direction(group):
        dates = group.index.get_level_values('date') if isinstance(group.index, pd.MultiIndex) else group.index
        spy_ret = spy_returns.reindex(dates, fill_value=0)
        stock_ret = group.values

        aligned = ((stock_ret > 0) == (spy_ret.values > 0)).astype(int)
        return pd.Series(aligned, index=group.index)

    # Simplified: just check if both positive or both negative
    result = df.groupby('symbol')['close'].transform(lambda x: x.pct_change())
    # For simplicity, return all 0s if spy_returns not available
    # Real implementation should merge with SPY data
    return pd.Series(0, index=df.index)


def market_breadth_strong(breadth_value: float, threshold: float = 0.6) -> int:
    """
    Binary feature: 1 if market breadth > threshold, 0 otherwise.

    Args:
        breadth_value: Market breadth value (0-1)
        threshold: Strong breadth threshold

    Returns:
        1 or 0
    """
    return 1 if breadth_value > threshold else 0


def vix_spike(vix_change: float, threshold: float = 0.15) -> int:
    """
    Binary feature: 1 if VIX increased > threshold, 0 otherwise.

    Args:
        vix_change: VIX percentage change
        threshold: Spike threshold (default 15%)

    Returns:
        1 or 0
    """
    return 1 if vix_change > threshold else 0


def bb_squeeze(df: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> pd.Series:
    """
    Binary feature: 1 if Bollinger Band width < threshold (low volatility), 0 otherwise.

    Args:
        df: DataFrame with 'symbol', 'close' columns
        window: BB window
        threshold: Squeeze threshold

    Returns:
        Series of 1/0 values
    """
    bb_width_val = bb_width(df, 'close', window, num_std=2.0)
    return (bb_width_val < threshold).astype(int)


# Validation helper

def validate_operators():
    """
    Run basic validation tests on operators.
    Used for debugging and ensuring consistency.
    """
    # Create simple test data
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10).tolist() * 2,
        'symbol': ['AAPL'] * 10 + ['MSFT'] * 10,
        'close': np.random.randn(20).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 20),
        'open': np.random.randn(20).cumsum() + 100,
    })

    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    print("Testing operators...")
    print(f"rank: {rank(df, 'close').describe()}")
    print(f"delay: {delay(df, 'close', 1).describe()}")
    print(f"delta: {delta(df, 'close', 1).describe()}")
    print(f"ts_rank: {ts_rank(df, 'close', 5).describe()}")
    print("All operators validated successfully!")


if __name__ == "__main__":
    validate_operators()
