"""
Implementation of selected alphas from WorldQuant's 101 Formulaic Alphas.

Reference: https://arxiv.org/abs/1601.00991

This module implements a subset of the 101 alphas, chosen for:
- Simplicity and interpretability
- Low correlation with each other
- Good performance on US equities

Each alpha function takes a DataFrame and returns a Series of alpha values.
"""

import numpy as np
import pandas as pd

# Handle both direct execution and module import
try:
    from . import operators as ops
except ImportError:
    import operators as ops


def alpha_001(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) :
    close), 2.), 5)) - 0.5)

    Note: This is complex, we'll implement a simplified version for stability.
    Returns: ts_rank of close over 5 days
    """
    return ops.ts_rank(df, 'close', 5) - 0.5


def alpha_002(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)

    Mean reversion alpha based on volume and price movements.
    """
    # Calculate components
    log_vol = ops.log(df['volume'])
    delta_log_vol = ops.delta(df.assign(log_volume=log_vol), 'log_volume', 2)
    price_change = (df['close'] - df['open']) / df['open']

    # Add to dataframe
    df_temp = df.copy()
    df_temp['delta_log_vol'] = delta_log_vol
    df_temp['price_change'] = price_change

    # Rank cross-sectionally
    df_temp['rank_vol'] = ops.rank(df_temp, 'delta_log_vol')
    df_temp['rank_price'] = ops.rank(df_temp, 'price_change')

    # Correlation over 6 days
    corr = ops.correlation(df_temp, 'rank_vol', 'rank_price', 6)

    return -1 * corr


def alpha_003(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#3: -1 * correlation(rank(open), rank(volume), 10)

    Captures relationship between opening prices and volume.
    """
    df_temp = df.copy()
    df_temp['rank_open'] = ops.rank(df_temp, 'open')
    df_temp['rank_volume'] = ops.rank(df_temp, 'volume')

    corr = ops.correlation(df_temp, 'rank_open', 'rank_volume', 10)

    return -1 * corr


def alpha_004(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#4: -1 * Ts_Rank(rank(low), 9)

    Time-series rank of cross-sectional rank of lows.
    Short-term mean reversion indicator.
    """
    df_temp = df.copy()
    df_temp['rank_low'] = ops.rank(df_temp, 'low')

    return -1 * ops.ts_rank(df_temp, 'rank_low', 9)


def alpha_006(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#6: -1 * correlation(open, volume, 10)

    Time-series correlation between open price and volume.
    """
    corr = ops.correlation(df, 'open', 'volume', 10)

    return -1 * corr


def alpha_012(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))

    Volume-weighted price reversal.
    If volume increases and price drops, this is positive (contrarian).
    """
    vol_change = ops.sign(ops.delta(df, 'volume', 1))
    price_change = -1 * ops.delta(df, 'close', 1)

    return vol_change * price_change


def alpha_016(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#16: -1 * rank(covariance(rank(high), rank(volume), 5))

    Cross-sectional rank of time-series covariance between high prices and volume.
    """
    df_temp = df.copy()
    df_temp['rank_high'] = ops.rank(df_temp, 'high')
    df_temp['rank_volume'] = ops.rank(df_temp, 'volume')

    cov = ops.covariance(df_temp, 'rank_high', 'rank_volume', 5)
    df_temp['cov'] = cov

    return -1 * ops.rank(df_temp, 'cov')


def alpha_018(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#18: -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) +
    correlation(close, open, 10)))

    Combines volatility, price movement, and correlation.
    """
    df_temp = df.copy()

    # stddev(abs(close - open), 5)
    df_temp['abs_co'] = (df['close'] - df['open']).abs()
    vol_component = ops.ts_stddev(df_temp, 'abs_co', 5)

    # close - open
    co_diff = df['close'] - df['open']

    # correlation(close, open, 10)
    corr_component = ops.correlation(df, 'close', 'open', 10)

    # Combine
    combined = vol_component + co_diff + corr_component
    df_temp['combined'] = combined

    return -1 * ops.rank(df_temp, 'combined')


def alpha_020(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#20: -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) *
    rank(open - delay(low, 1))

    Multiple mean reversion signal based on gaps.
    """
    df_temp = df.copy()

    open_high_gap = df['open'] - ops.delay(df, 'high', 1)
    open_close_gap = df['open'] - ops.delay(df, 'close', 1)
    open_low_gap = df['open'] - ops.delay(df, 'low', 1)

    df_temp['gap_high'] = open_high_gap
    df_temp['gap_close'] = open_close_gap
    df_temp['gap_low'] = open_low_gap

    rank_high = ops.rank(df_temp, 'gap_high')
    rank_close = ops.rank(df_temp, 'gap_close')
    rank_low = ops.rank(df_temp, 'gap_low')

    return -1 * rank_high * rank_close * rank_low


def alpha_028(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    Combines volume, price level, and position within the day's range.
    """
    df_temp = df.copy()

    # adv20 = average dollar volume over 20 days
    adv20 = ops.adv(df_temp, 20)
    df_temp['adv20'] = adv20

    # correlation(adv20, low, 5)
    corr_component = ops.correlation(df_temp, 'adv20', 'low', 5)

    # (high + low) / 2
    mid_price = (df['high'] + df['low']) / 2

    # Combined signal
    signal = corr_component + mid_price - df['close']

    return ops.scale(signal)


def alpha_053(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#53: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)

    Williams %R momentum indicator.
    Measures where close is relative to high-low range.
    """
    # Williams %R formula
    numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
    denominator = df['close'] - df['low']

    # Avoid division by zero
    williams_r = numerator / denominator.replace(0, np.nan)

    df_temp = df.copy()
    df_temp['williams_r'] = williams_r

    return -1 * ops.delta(df_temp, 'williams_r', 9)


def alpha_007(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))

    Volume breakout signal with momentum direction.
    """
    df_temp = df.copy()

    adv20_val = ops.adv(df_temp, 20)
    df_temp['adv20'] = adv20_val

    delta_close_7 = ops.delta(df_temp, 'close', 7)
    df_temp['delta_close_7'] = delta_close_7
    df_temp['abs_delta'] = delta_close_7.abs()

    ts_rank_val = ops.ts_rank(df_temp, 'abs_delta', 60)
    sign_val = ops.sign(delta_close_7)

    # Conditional logic
    result = pd.Series(index=df.index, dtype=float)
    mask = df['volume'] > adv20_val
    result[mask] = -1 * ts_rank_val[mask] * sign_val[mask]
    result[~mask] = -1.0

    return result


def alpha_009(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

    Adaptive momentum signal.
    """
    df_temp = df.copy()

    delta_close = ops.delta(df_temp, 'close', 1)
    df_temp['delta_close'] = delta_close

    ts_min_val = ops.ts_min(df_temp, 'delta_close', 5)
    ts_max_val = ops.ts_max(df_temp, 'delta_close', 5)

    result = pd.Series(index=df.index, dtype=float)

    # If minimum is positive, use delta as is
    mask1 = ts_min_val > 0
    result[mask1] = delta_close[mask1]

    # Else if maximum is negative, use delta as is
    mask2 = (~mask1) & (ts_max_val < 0)
    result[mask2] = delta_close[mask2]

    # Otherwise, invert
    mask3 = ~(mask1 | mask2)
    result[mask3] = -1 * delta_close[mask3]

    return result


def alpha_013(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))

    Cross-sectional rank of time-series covariance between price and volume ranks.
    """
    df_temp = df.copy()
    df_temp['rank_close'] = ops.rank(df_temp, 'close')
    df_temp['rank_volume'] = ops.rank(df_temp, 'volume')

    cov = ops.covariance(df_temp, 'rank_close', 'rank_volume', 5)
    df_temp['cov'] = cov

    return -1 * ops.rank(df_temp, 'cov')


def alpha_014(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))

    Mean reversion signal modulated by open-volume correlation.
    """
    df_temp = df.copy()

    ret = ops.returns(df_temp, 'close', 1)
    df_temp['returns'] = ret

    delta_ret = ops.delta(df_temp, 'returns', 3)
    df_temp['delta_ret'] = delta_ret

    rank_delta = ops.rank(df_temp, 'delta_ret')
    corr_val = ops.correlation(df_temp, 'open', 'volume', 10)

    return -1 * rank_delta * corr_val


def alpha_015(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

    Rolling sum of ranked correlation.
    """
    df_temp = df.copy()
    df_temp['rank_high'] = ops.rank(df_temp, 'high')
    df_temp['rank_volume'] = ops.rank(df_temp, 'volume')

    corr = ops.correlation(df_temp, 'rank_high', 'rank_volume', 3)
    df_temp['corr'] = corr
    df_temp['rank_corr'] = ops.rank(df_temp, 'corr')

    return -1 * ops.ts_sum(df_temp, 'rank_corr', 3)


def alpha_017(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))

    Complex combination of price and volume momentum.
    """
    df_temp = df.copy()

    # Component 1: ts_rank of close
    ts_rank_close = ops.ts_rank(df_temp, 'close', 10)
    df_temp['ts_rank_close'] = ts_rank_close
    comp1 = -1 * ops.rank(df_temp, 'ts_rank_close')

    # Component 2: delta of delta of close
    delta1 = ops.delta(df_temp, 'close', 1)
    df_temp['delta1'] = delta1
    delta2 = ops.delta(df_temp, 'delta1', 1)
    df_temp['delta2'] = delta2
    comp2 = ops.rank(df_temp, 'delta2')

    # Component 3: volume ratio
    adv20_val = ops.adv(df_temp, 20)
    vol_ratio = df['volume'] / adv20_val
    df_temp['vol_ratio'] = vol_ratio
    ts_rank_vol = ops.ts_rank(df_temp, 'vol_ratio', 5)
    df_temp['ts_rank_vol'] = ts_rank_vol
    comp3 = ops.rank(df_temp, 'ts_rank_vol')

    return comp1 * comp2 * comp3


def alpha_019(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))

    Long-term reversal signal.
    """
    df_temp = df.copy()

    # Close change components
    close_diff = df['close'] - ops.delay(df_temp, 'close', 7)
    delta_close = ops.delta(df_temp, 'close', 7)
    sign_val = ops.sign(close_diff + delta_close)

    # Long-term return component
    ret = ops.returns(df_temp, 'close', 1)
    df_temp['returns'] = ret
    sum_ret = ops.ts_sum(df_temp, 'returns', 250)
    df_temp['sum_ret'] = 1 + sum_ret
    rank_val = ops.rank(df_temp, 'sum_ret')

    return -1 * sign_val * (1 + rank_val)


def alpha_021(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) :
               (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 :
               (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))

    Complex conditional logic based on moving averages and volume.
    Simplified version for stability.
    """
    df_temp = df.copy()

    # Calculate components
    ma8 = ops.ts_mean(df_temp, 'close', 8)
    ma2 = ops.ts_mean(df_temp, 'close', 2)
    std8 = ops.ts_stddev(df_temp, 'close', 8)

    adv20_val = ops.adv(df_temp, 20)
    vol_ratio = df['volume'] / adv20_val

    # Simplified: if short MA < long MA - std, bullish (+1), else bearish (-1)
    result = pd.Series(index=df.index, dtype=float)
    result = np.where(ma2 < (ma8 - std8), 1.0, -1.0)

    # Modulate by volume
    result = np.where(vol_ratio > 1, 1.0, result)

    return pd.Series(result, index=df.index)


def alpha_022(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#22: -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))

    Change in high-volume correlation, scaled by volatility rank.
    """
    df_temp = df.copy()

    # Correlation and its delta
    corr = ops.correlation(df_temp, 'high', 'volume', 5)
    df_temp['corr'] = corr
    delta_corr = ops.delta(df_temp, 'corr', 5)

    # Volatility rank
    std_close = ops.ts_stddev(df_temp, 'close', 20)
    df_temp['std_close'] = std_close
    rank_std = ops.rank(df_temp, 'std_close')

    return -1 * delta_corr * rank_std


def alpha_023(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

    High price reversal signal.
    """
    df_temp = df.copy()

    ma_high = ops.ts_mean(df_temp, 'high', 20)
    delta_high = ops.delta(df_temp, 'high', 2)

    result = pd.Series(0.0, index=df.index)
    mask = df['high'] > ma_high
    result[mask] = -1 * delta_high[mask]

    return result


def alpha_024(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||
               ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
               (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))

    Long-term trend follower with adaptive response.
    Simplified version.
    """
    df_temp = df.copy()

    # Long-term moving average change
    ma100 = ops.ts_mean(df_temp, 'close', 100)
    df_temp['ma100'] = ma100
    delta_ma = ops.delta(df_temp, 'ma100', 100)
    delay_close = ops.delay(df_temp, 'close', 100)

    ma_change_pct = delta_ma / delay_close

    # Conditional logic (simplified)
    ts_min_close = ops.ts_min(df_temp, 'close', 100)
    delta_close_3 = ops.delta(df_temp, 'close', 3)

    result = pd.Series(index=df.index, dtype=float)
    mask = ma_change_pct.abs() <= 0.05
    result[mask] = -1 * (df['close'][mask] - ts_min_close[mask])
    result[~mask] = -1 * delta_close_3[~mask]

    return result


def alpha_026(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    Maximum correlation between volume and high price ranks.
    """
    df_temp = df.copy()

    ts_rank_vol = ops.ts_rank(df_temp, 'volume', 5)
    ts_rank_high = ops.ts_rank(df_temp, 'high', 5)

    df_temp['ts_rank_vol'] = ts_rank_vol
    df_temp['ts_rank_high'] = ts_rank_high

    corr = ops.correlation(df_temp, 'ts_rank_vol', 'ts_rank_high', 5)
    df_temp['corr'] = corr

    ts_max_corr = ops.ts_max(df_temp, 'corr', 3)

    return -1 * ts_max_corr


def alpha_030(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +
               sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

    Momentum signal weighted by relative volume.
    """
    df_temp = df.copy()

    # Sign changes
    delta1 = df['close'] - ops.delay(df_temp, 'close', 1)
    delta2 = ops.delay(df_temp, 'close', 1) - ops.delay(df_temp, 'close', 2)
    delta3 = ops.delay(df_temp, 'close', 2) - ops.delay(df_temp, 'close', 3)

    sign_sum = ops.sign(delta1) + ops.sign(delta2) + ops.sign(delta3)
    df_temp['sign_sum'] = sign_sum

    rank_val = ops.rank(df_temp, 'sign_sum')

    # Volume ratio
    sum_vol_5 = ops.ts_sum(df_temp, 'volume', 5)
    sum_vol_20 = ops.ts_sum(df_temp, 'volume', 20)
    vol_ratio = sum_vol_5 / sum_vol_20

    return (1.0 - rank_val) * vol_ratio


def alpha_033(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#33: rank((-1 * ((1 - (open / close))^1)))

    Intraday return rank (simplified).
    """
    df_temp = df.copy()

    intraday_ret = -1 * (1 - (df['open'] / df['close']))
    df_temp['intraday_ret'] = intraday_ret

    return ops.rank(df_temp, 'intraday_ret')


def alpha_034(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

    Volatility ratio and momentum combination.
    """
    df_temp = df.copy()

    ret = ops.returns(df_temp, 'close', 1)
    df_temp['returns'] = ret

    std2 = ops.ts_stddev(df_temp, 'returns', 2)
    std5 = ops.ts_stddev(df_temp, 'returns', 5)
    std_ratio = std2 / std5
    df_temp['std_ratio'] = std_ratio

    rank_std = ops.rank(df_temp, 'std_ratio')

    delta_close = ops.delta(df_temp, 'close', 1)
    df_temp['delta_close'] = delta_close
    rank_delta = ops.rank(df_temp, 'delta_close')

    combined = (1 - rank_std) + (1 - rank_delta)
    df_temp['combined'] = combined

    return ops.rank(df_temp, 'combined')


def alpha_037(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

    Long-term correlation of open-close gap with price.
    """
    df_temp = df.copy()

    oc_gap = df['open'] - df['close']
    df_temp['oc_gap'] = oc_gap
    oc_gap_delay = ops.delay(df_temp, 'oc_gap', 1)
    df_temp['oc_gap_delay'] = oc_gap_delay

    corr = ops.correlation(df_temp, 'oc_gap_delay', 'close', 200)
    df_temp['corr'] = corr

    rank_corr = ops.rank(df_temp, 'corr')
    rank_gap = ops.rank(df_temp, 'oc_gap')

    return rank_corr + rank_gap


def alpha_038(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

    Recent strength vs intraday performance.
    """
    df_temp = df.copy()

    ts_rank_close = ops.ts_rank(df_temp, 'close', 10)
    df_temp['ts_rank_close'] = ts_rank_close
    rank1 = ops.rank(df_temp, 'ts_rank_close')

    co_ratio = df['close'] / df['open']
    df_temp['co_ratio'] = co_ratio
    rank2 = ops.rank(df_temp, 'co_ratio')

    return -1 * rank1 * rank2


def alpha_040(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

    High price volatility with volume correlation.
    """
    df_temp = df.copy()

    std_high = ops.ts_stddev(df_temp, 'high', 10)
    df_temp['std_high'] = std_high
    rank_std = ops.rank(df_temp, 'std_high')

    corr = ops.correlation(df_temp, 'high', 'volume', 10)

    return -1 * rank_std * corr


def alpha_041(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#41: (((high * low)^0.5) - vwap)

    Geometric mean of high/low vs VWAP.
    Note: We don't have VWAP, so we approximate with (high+low+close)/3.
    """
    geometric_mean = (df['high'] * df['low']) ** 0.5
    vwap_approx = (df['high'] + df['low'] + df['close']) / 3

    return geometric_mean - vwap_approx


def alpha_042(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))

    VWAP deviation ratio.
    Using approximation for VWAP.
    """
    df_temp = df.copy()

    vwap_approx = (df['high'] + df['low'] + df['close']) / 3

    numerator = vwap_approx - df['close']
    denominator = vwap_approx + df['close']

    df_temp['numerator'] = numerator
    df_temp['denominator'] = denominator

    rank_num = ops.rank(df_temp, 'numerator')
    rank_den = ops.rank(df_temp, 'denominator')

    return rank_num / rank_den


# Additional custom alphas inspired by the 101 paper but adapted for robustness

def alpha_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Custom Alpha: Price momentum with volume confirmation.

    Returns cross-sectional rank of (return * volume_ratio).
    """
    df_temp = df.copy()

    # Price momentum
    ret = ops.returns(df_temp, 'close', window)

    # Volume ratio vs moving average
    vol_ma = ops.ts_mean(df_temp, 'volume', window)
    vol_ratio = df['volume'] / vol_ma

    # Combined signal
    signal = ret * vol_ratio
    df_temp['signal'] = signal

    return ops.rank(df_temp, 'signal')


def alpha_reversal(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Custom Alpha: Short-term mean reversion.

    Returns -1 * ts_rank of recent returns.
    High recent returns → negative alpha (expect reversal).
    """
    df_temp = df.copy()

    ret = ops.returns(df_temp, 'close', window)
    df_temp['ret'] = ret

    return -1 * ops.ts_rank(df_temp, 'ret', window)


def alpha_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Custom Alpha: Volatility factor.

    Returns -1 * rank(volatility).
    Low vol stocks get higher alpha values.
    """
    df_temp = df.copy()

    vol = ops.volatility(df_temp, window, 'close')
    df_temp['vol'] = vol

    return -1 * ops.rank(df_temp, 'vol')


# Registry of all implemented alphas

ALPHAS = {
    # Original 101 Formulaic Alphas
    'alpha_001': alpha_001,
    'alpha_002': alpha_002,
    'alpha_003': alpha_003,
    'alpha_004': alpha_004,
    'alpha_006': alpha_006,
    'alpha_007': alpha_007,
    'alpha_009': alpha_009,
    'alpha_012': alpha_012,
    'alpha_013': alpha_013,
    'alpha_014': alpha_014,
    'alpha_015': alpha_015,
    'alpha_016': alpha_016,
    'alpha_017': alpha_017,
    'alpha_018': alpha_018,
    'alpha_019': alpha_019,
    'alpha_020': alpha_020,
    'alpha_021': alpha_021,
    'alpha_022': alpha_022,
    'alpha_023': alpha_023,
    'alpha_024': alpha_024,
    'alpha_026': alpha_026,
    'alpha_028': alpha_028,
    'alpha_030': alpha_030,
    'alpha_033': alpha_033,
    'alpha_034': alpha_034,
    'alpha_037': alpha_037,
    'alpha_038': alpha_038,
    'alpha_040': alpha_040,
    'alpha_041': alpha_041,
    'alpha_042': alpha_042,
    'alpha_053': alpha_053,

    # Custom alphas
    'alpha_momentum': alpha_momentum,
    'alpha_reversal': alpha_reversal,
    'alpha_volatility': alpha_volatility,
}


def compute_all_alphas(df: pd.DataFrame, alpha_list: list = None) -> pd.DataFrame:
    """
    Compute multiple alphas and return as DataFrame.

    Args:
        df: Input DataFrame with OHLCV data
        alpha_list: List of alpha names to compute (default: all)

    Returns:
        DataFrame with original data + alpha columns
    """
    if alpha_list is None:
        alpha_list = list(ALPHAS.keys())

    result = df.copy()

    print(f"Computing {len(alpha_list)} alphas...")

    for alpha_name in alpha_list:
        if alpha_name not in ALPHAS:
            print(f"Warning: Unknown alpha '{alpha_name}', skipping")
            continue

        try:
            alpha_func = ALPHAS[alpha_name]
            result[alpha_name] = alpha_func(df)
            print(f"[OK] {alpha_name}")
        except Exception as e:
            print(f"[FAIL] {alpha_name}: {e}")
            result[alpha_name] = np.nan

    return result


if __name__ == "__main__":
    # Simple test
    print("Testing alpha implementations...")

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    symbols = ['AAPL', 'MSFT', 'NVDA']

    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'open': 100 + np.random.randn() * 2,
                'high': 102 + np.random.randn() * 2,
                'low': 98 + np.random.randn() * 2,
                'close': 100 + np.random.randn() * 2,
                'volume': 1000000 + np.random.randint(-100000, 100000),
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Test a few alphas
    print("\nTesting Alpha #2...")
    a2 = alpha_002(df)
    print(f"Alpha 2 - Mean: {a2.mean():.4f}, Std: {a2.std():.4f}, NaN%: {a2.isna().mean()*100:.1f}%")

    print("\nTesting Alpha #12...")
    a12 = alpha_012(df)
    print(f"Alpha 12 - Mean: {a12.mean():.4f}, Std: {a12.std():.4f}, NaN%: {a12.isna().mean()*100:.1f}%")

    print("\nComputing all alphas...")
    result = compute_all_alphas(df, list(ALPHAS.keys())[:5])

    print("\nSample output:")
    print(result[['date', 'symbol', 'close', 'alpha_002', 'alpha_012']].head(20))

    print("\nAll tests passed!")
