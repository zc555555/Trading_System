"""
MULTI-FACTOR MODEL - FACTOR DEFINITIONS

Organize 80 features into 5 major factor categories:
1. Momentum - Price momentum and rate of change
2. Trend - Trend following and moving averages
3. Volatility - Volatility and price range
4. Volume - Trading volume patterns
5. Market - Market environment and correlations

Each factor will have its own ensemble model.
"""

# ============================================================================
# FACTOR 1: MOMENTUM (Price momentum and acceleration)
#
# W1.6 outcome: tested adding 20 xs features to this factor and got IC drop
# from 0.2609 -> 0.2577. At 27 features the model was already saturated for
# this 600k-sample dataset; more features just added noise. KEEPING the
# original 27-feature definition. (Cross-sectional return ranks are still
# available as columns -- they just aren't fed to this factor.)
# ============================================================================
MOMENTUM_FEATURES = [
    # Returns across timeframes
    'returns_1d_positive',
    'returns_2d',
    'returns_3d',
    'returns_5d',
    'returns_7d',
    'returns_10d',
    'returns_15d',
    'returns_20d',
    'returns_30d',
    'returns_60d',
    'momentum_50d',

    # Rate of change
    'roc_3d',
    'roc_7d',
    'roc_15d',
    'roc_30d',

    # Consecutive movements
    'consecutive_up_days',
    'consecutive_down_days',

    # Momentum oscillators
    'macd',
    'macd_hist',
    'macd_positive',
    'rsi_7',
    'rsi_oversold',
    'cci_20',
    'williams_r_14',
    'trix_14',
    'dpo_20',
    'mfi_14',
]

# ============================================================================
# FACTOR 2: TREND (cross-sectional trend strength)
#
# W1.6 (2026-05) rewrite:
#   Replaced absolute / binary trend features with cross-sectional z-scores.
#   The old definition mixed binary flags (golden_cross, price_above_sma_20)
#   that just split the universe 50/50 with absolute MA-distance features that
#   the model could not rank against peers. Result: trend IC was -0.012 on a
#   302-stock universe -- worse than random.
#
#   The new definition asks "how does this stock RANK today on trend-distance
#   from peers?" which is what cross-sectional alpha actually means.
# ============================================================================
TREND_FEATURES = [
    # Price-vs-moving-average, ranked cross-sectionally per day
    'price_vs_ma_3d_xs',
    'price_vs_ma_7d_xs',
    'price_vs_ma_15d_xs',
    'price_vs_ma_30d_xs',

    # MACD ranked cross-sectionally
    'macd_xs',
    'macd_hist_xs',
]

# ============================================================================
# FACTOR 3: VOLATILITY (per-stock volatility + cross-sectional vol rank)
#
# W1.6 (2026-05): added xs versions of volatility timeframes so the model
# can also pick up "low-vol stocks today" (the low-vol anomaly --
# Frazzini-Pedersen 2014 -- where low-beta low-vol names tend to outperform
# on a risk-adjusted basis).
# ============================================================================
VOLATILITY_FEATURES = [
    # ---- absolute per-stock volatility (W1 originals) ----
    'volatility_3d',
    'volatility_10d',
    'volatility_20d',
    'volatility_60d',

    'bb_width',
    'bb_squeeze',
    'kc_position',

    # ---- W1.6 cross-sectional volatility ranks ----
    'volatility_3d_xs',
    'volatility_10d_xs',
    'volatility_20d_xs',
    'volatility_60d_xs',
    'bb_width_xs',
]

# ============================================================================
# FACTOR 4: VOLUME (cross-sectional volume anomalies)
#
# W1.6 (2026-05) rewrite:
#   Same logic as TREND -- old definition used per-stock volume_ratio which
#   measures the stock against ITS OWN history. For cross-sectional ranking we
#   need to know how this stock's volume burst compares to the rest of the
#   universe TODAY. Old IC: -0.016; binary volume_increasing was pure noise.
# ============================================================================
VOLUME_FEATURES = [
    # Volume ratios ranked cross-sectionally per day
    'volume_ratio_3d_xs',
    'volume_ratio_5d_xs',
    'volume_ratio_7d_xs',
    'volume_ratio_15d_xs',
    'volume_ratio_20d_xs',
    'volume_ratio_30d_xs',

    # Volume volatility ranked
    'volume_std_20d_xs',

    # VWAP deviation ranked
    'vwap_ratio_xs',
]

# ============================================================================
# FACTOR 5: MARKET (Market environment and macro factors)
# ============================================================================
MARKET_FEATURES = [
    # SPY (S&P 500)
    'spy_returns_1d_y',
    'spy_returns_5d_y',
    'spy_returns_20d_y',
    'spy_volatility_20d_y',

    # QQQ (Nasdaq)
    'qqq_returns_1d_y',
    'qqq_returns_5d_y',

    # DIA (Dow Jones)
    'dia_returns_1d_y',
    'dia_returns_5d_y',

    # IWM (Small caps)
    'iwm_returns_1d_y',
    'iwm_returns_5d_y',

    # GLD (Gold)
    'gld_returns_1d_y',
    'gld_returns_5d_y',

    # TLT (Bonds)
    'tlt_returns_1d_y',
    'tlt_returns_5d_y',

    # USO (Oil)
    'uso_returns_1d_y',
    'uso_returns_5d_y',

    # UUP (Dollar)
    'uup_returns_1d_y',
    'uup_returns_5d_y',

    # VIX (Volatility index)
    'vix_level_y',
    'vix_change_1d_y',
    'vix_change_5d_y',
]

# ============================================================================
# FACTOR 6: ALPHA (Proprietary alpha factors)
# ============================================================================
ALPHA_FEATURES = [
    'alpha_019',
    'alpha_041',
    'alpha_042',
]

# 52-week-high anchoring (George & Hwang 2004). CONDITIONALLY ADOPTED
# 2026-08-10 (hypothesis #15 in evaluation/results/hypothesis_ledger.csv):
# dev blend IC 0.0129 -> 0.0314, holdout t=2.88 vs a Sidak family bar of
# 2.93 -- below family-wise significance, adopted on the dev rule with a
# pre-registered removal trigger: negative fresh tier at the ~2026-11
# quarterly review removes it.
HIGH52_FEATURES = [
    'pct_52w_high',
    'days_from_high',
    'pct_52w_high_xs',
    'days_from_high_xs',
]

# ============================================================================
# ALL FACTORS DICTIONARY
# ============================================================================
FACTOR_GROUPS = {
    'momentum': MOMENTUM_FEATURES,
    'trend': TREND_FEATURES,
    'volatility': VOLATILITY_FEATURES,
    'volume': VOLUME_FEATURES,
    'market': MARKET_FEATURES,
    'alpha': ALPHA_FEATURES,
    'high52': HIGH52_FEATURES,
}

# ============================================================================
# DEFAULT FACTOR WEIGHTS (static FALLBACK only -- production uses the
# IC-based resolver; see factors/factor_weighting.py)
# ============================================================================
FACTOR_WEIGHTS = {
    'momentum': 0.23,      # Short-term price momentum
    'trend': 0.19,         # Trend following
    'volatility': 0.10,    # Volatility patterns
    'volume': 0.09,        # Volume confirmation
    'market': 0.24,        # Market environment
    'alpha': 0.10,         # Proprietary factors
    'high52': 0.05,        # 52w-high anchoring (conditional, 2026-08)
}

# Agent-mined factors adopted under rulebook track B (factors/mined_factors.json).
# Each gets the same 5% static-fallback weight high52 got; the static dict is
# renormalised. Production weights come from the IC resolver anyway.
try:
    from .mined_factors import mined_factor_groups as _mined_factor_groups
except ImportError:
    from mined_factors import mined_factor_groups as _mined_factor_groups
_MINED_GROUPS = _mined_factor_groups()
if _MINED_GROUPS:
    FACTOR_GROUPS.update(_MINED_GROUPS)
    _scale = 1.0 - 0.05 * len(_MINED_GROUPS)
    FACTOR_WEIGHTS = {f: w * _scale for f, w in FACTOR_WEIGHTS.items()}
    FACTOR_WEIGHTS.update({f: 0.05 for f in _MINED_GROUPS})

# Verify weights sum to 1.0
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 0.001, "Factor weights must sum to 1.0"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_features():
    """Get list of all features across all factors."""
    all_features = []
    for features in FACTOR_GROUPS.values():
        all_features.extend(features)
    return all_features


def verify_feature_coverage(available_features):
    """
    Verify that all defined factors are available in the dataset.

    Args:
        available_features: List of features available in the data

    Returns:
        dict: Report of missing features per factor
    """
    report = {}

    for factor_name, factor_features in FACTOR_GROUPS.items():
        missing = [f for f in factor_features if f not in available_features]
        available = [f for f in factor_features if f in available_features]

        report[factor_name] = {
            'total': len(factor_features),
            'available': len(available),
            'missing': len(missing),
            'missing_features': missing,
            'coverage_pct': len(available) / len(factor_features) * 100
        }

    return report


def print_factor_summary():
    """Print a summary of all factor definitions."""
    print("\n" + "="*80)
    print("MULTI-FACTOR MODEL - FACTOR SUMMARY")
    print("="*80)

    total_features = 0

    for factor_name, features in FACTOR_GROUPS.items():
        weight = FACTOR_WEIGHTS[factor_name]
        print(f"\n{factor_name.upper()}: {len(features)} features (weight: {weight*100:.0f}%)")
        print("-" * 80)
        for i, feat in enumerate(features, 1):
            print(f"  {i:2}. {feat}")
        total_features += len(features)

    print("\n" + "="*80)
    print(f"TOTAL: {total_features} features across {len(FACTOR_GROUPS)} factors")
    print("="*80)


if __name__ == "__main__":
    print_factor_summary()

    # Verify all 80 features are accounted for
    all_feats = get_all_features()
    print(f"\nTotal features defined: {len(all_feats)}")
    print(f"Unique features: {len(set(all_feats))}")

    if len(all_feats) != len(set(all_feats)):
        print("\n[WARNING] Duplicate features found!")
        from collections import Counter
        counts = Counter(all_feats)
        dups = [f for f, c in counts.items() if c > 1]
        print(f"Duplicates: {dups}")
