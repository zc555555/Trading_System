"""
MULTI-FACTOR DAILY TRADING SIGNALS

Generate trading signals using multi-factor ensemble with 5-day weighted prediction.

Each factor (Momentum, Trend, Volatility, Volume, Market, Alpha) makes independent
predictions, then they are combined with configurable weights.

Improvement over single model:
- Factor-specific expertise
- Transparent factor contribution analysis
- Easier to diagnose and improve
"""

import io
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Windows: piped/redirected stdout defaults to cp1252, which cannot encode
# the Chinese status lines (finbert_sentiment etc.) and crashes the script
# mid-run under Task Scheduler. Force UTF-8, never crash on odd characters.
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import factor definitions
# W1.5: FACTOR_WEIGHTS is resolved at import time via load_effective_factor_weights()
# (default strategy = ic_proportional), so anti-predictive factors get zero weight.
from factors.factor_definitions import FACTOR_GROUPS
from factors.factor_weighting import load_effective_factor_weights

# Resolve effective weights immediately. Helpers below read this at call time,
# so re-resolving inside main() is unnecessary.
FACTOR_WEIGHTS = load_effective_factor_weights()

from config_trend_filters import (
    FILTER_7D_MIN_RETURN,
    FILTER_14D_MIN_RETURN,
    FILTER_30D_MIN_RETURN,
    FILTER_TREND_STRENGTH_MIN,
    REQUIRE_PRICE_ABOVE_MAs
)

# Add parent directory to path for news sentiment import
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from finbert_sentiment import FinBERTSentimentAnalyzer as NewsSentimentAnalyzer


def calculate_win_rates_multi_factor(factor_ensembles, df_all):
    """Calculate historical win rates using multi-factor predictions.

    W2-A: per-date z-scores each factor's prediction so multi-horizon factors
    blend on a comparable scale. The blended signal is then compared against
    future_return (1d). Reported win rate is for the RANK direction at 1d
    horizon; the underlying factor models may predict at longer horizons but
    correlation of the ranking with next-day direction is what we check here.
    """
    try:
        df_recent = df_all[df_all['label'].notna()].copy()
        df_recent = df_recent.tail(500)

        if len(df_recent) < 100:
            return None

        # Generate multi-factor predictions
        factor_predictions = {}

        for factor_name, ensemble_data in factor_ensembles.items():
            models = ensemble_data['models']
            weights = ensemble_data['weights']
            feature_cols = ensemble_data['feature_cols']

            X = df_recent[feature_cols].values
            X = pd.DataFrame(X, columns=feature_cols).ffill().fillna(0).values

            # Ensemble prediction for this factor
            predictions = {}
            for model_name in sorted(models.keys()):
                pred = models[model_name].predict(X)
                predictions[model_name] = pred

            pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
            weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
            factor_pred = pred_matrix @ weights_array

            # W2-A: per-date z-score so factors trained on different horizons mix fairly
            tmp = pd.DataFrame({
                'date': df_recent['date'].values,
                'p': factor_pred,
            })
            tmp['p_z'] = tmp.groupby('date')['p'].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
            )
            factor_predictions[factor_name] = tmp['p_z'].values

        # Combine factors with weights
        final_pred = sum(
            factor_predictions[name] * FACTOR_WEIGHTS[name]
            for name in factor_predictions.keys()
        )

        # Get actual returns
        y_true = df_recent['future_return'].values

        # Calculate win rates
        buy_mask = final_pred > 0
        sell_mask = final_pred < 0

        buy_correct = ((final_pred > 0) & (y_true > 0)).sum()
        buy_total = buy_mask.sum()
        buy_win_rate = buy_correct / buy_total if buy_total > 0 else 0

        sell_correct = ((final_pred < 0) & (y_true < 0)).sum()
        sell_total = sell_mask.sum()
        sell_win_rate = sell_correct / sell_total if sell_total > 0 else 0

        overall_correct = ((final_pred > 0) == (y_true > 0)).sum()
        overall_win_rate = overall_correct / len(y_true)

        return {
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'overall_win_rate': overall_win_rate,
            'buy_total': int(buy_total),
            'sell_total': int(sell_total),
            'sample_size': len(df_recent)
        }
    except Exception as e:
        print(f"[WARNING] Could not calculate win rates: {e}")
        return None


def _per_date_zscore(values: np.ndarray) -> np.ndarray:
    """Standardize a 1-D prediction array to mean 0 / std 1.

    Used inside the per-date loop so each factor's contribution is on a
    comparable scale before factor-level blending. Necessary after W2-A
    because factors trained at different horizons (1d, 5d, 20d) produce
    predictions on vastly different magnitudes -- without normalization a
    factor weighted 10% but predicting at 20d horizon dominates a factor
    weighted 40% but predicting at 1d horizon.
    """
    arr = np.asarray(values, dtype=float)
    sigma = arr.std()
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sigma


def get_weighted_predictions_multi_factor(df, factor_ensembles, n_days=5):
    """
    Generate multi-factor predictions with 5-day weighting.

    For each factor:
    1. Generate predictions for last 5 days
    2. Apply time weights (30%, 25%, 20%, 15%, 10%)
    3. Combine factors with factor weights
    """
    day_weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # newest to oldest

    dates = sorted(df['date'].unique())

    if len(dates) < n_days:
        print(f"[WARNING] Only {len(dates)} days available, need {n_days}")
        n_days = len(dates)
        day_weights = day_weights[-n_days:]
        day_weights = [w / sum(day_weights) for w in day_weights]

    last_n_dates = dates[-n_days:]

    print(f"\n[INFO] Using last {n_days} days for weighted prediction:")
    for i, date in enumerate(reversed(last_n_dates)):
        days_ago = n_days - i - 1
        weight = day_weights[days_ago]
        print(f"  {date.date()}: {weight*100:.0f}% weight ({days_ago} days ago)")

    # Collect daily predictions for each factor
    daily_factor_predictions = {factor_name: [] for factor_name in factor_ensembles.keys()}

    for date in last_n_dates:
        date_data = df[df['date'] == date].copy()

        if len(date_data) == 0:
            continue

        # Generate predictions for each factor
        for factor_name, ensemble_data in factor_ensembles.items():
            models = ensemble_data['models']
            weights = ensemble_data['weights']
            feature_cols = ensemble_data['feature_cols']

            X = date_data[feature_cols].values
            X = pd.DataFrame(X, columns=feature_cols).ffill().fillna(0).values

            # Ensemble prediction for this factor
            predictions = {}
            for model_name in sorted(models.keys()):
                pred = models[model_name].predict(X)
                predictions[model_name] = pred

            pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
            weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
            factor_pred = pred_matrix @ weights_array

            # W2-A: per-date z-score so multi-horizon factors blend fairly.
            factor_pred = _per_date_zscore(factor_pred)

            # Store predictions with symbol info
            date_data = date_data.copy()
            date_data[f'{factor_name}_pred'] = factor_pred
            daily_factor_predictions[factor_name].append(date_data)

    # Combine all predictions
    all_factor_preds = {}
    for factor_name, daily_preds in daily_factor_predictions.items():
        if daily_preds:
            all_factor_preds[factor_name] = pd.concat(daily_preds, ignore_index=True)

    # Calculate weighted average per symbol for each factor
    weighted_predictions = []

    # Get symbols from first factor
    first_factor = list(all_factor_preds.values())[0]
    symbols = first_factor['symbol'].unique()

    for symbol in symbols:
        # Get latest row for this symbol
        symbol_data = first_factor[first_factor['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        latest_row = symbol_data.iloc[-1].copy()

        # Calculate weighted prediction for each factor
        factor_contributions = {}

        for factor_name, factor_df in all_factor_preds.items():
            symbol_factor_data = factor_df[factor_df['symbol'] == symbol].copy()
            symbol_factor_data = symbol_factor_data.sort_values('date')

            preds = symbol_factor_data[f'{factor_name}_pred'].values

            # Apply day weights
            if len(preds) == n_days:
                weighted_pred = sum(p * w for p, w in zip(preds, reversed(day_weights)))
            else:
                available_weights = day_weights[-len(preds):]
                available_weights = [w / sum(available_weights) for w in available_weights]
                weighted_pred = sum(p * w for p, w in zip(preds, reversed(available_weights)))

            factor_contributions[factor_name] = weighted_pred

        # Combine factors with factor weights
        final_prediction = sum(
            factor_contributions[name] * FACTOR_WEIGHTS[name]
            for name in factor_contributions.keys()
        )

        latest_row['prediction'] = final_prediction
        latest_row['confidence'] = abs(final_prediction)

        # Store factor contributions for analysis
        for name, contrib in factor_contributions.items():
            latest_row[f'factor_{name}'] = contrib

        weighted_predictions.append(latest_row)

    result_df = pd.DataFrame(weighted_predictions)

    # Persist the FULL-universe per-factor scores for the IC decay monitor
    # (evaluation/ic_monitor.py). Each factor's realized 20-day rank IC is
    # computed from these once the labels mature; without this dump the
    # nightly scores exist only inside this process. Never fatal.
    try:
        _scores_dir = Path(__file__).resolve().parent / "artifacts" / "factor_scores"
        _scores_dir.mkdir(parents=True, exist_ok=True)
        _score_cols = (["date", "symbol"]
                       + [c for c in result_df.columns if c.startswith("factor_")]
                       + ["prediction"])
        _d = pd.to_datetime(result_df["date"]).max()
        result_df[_score_cols].to_parquet(
            _scores_dir / f"factor_scores_{_d.strftime('%Y%m%d')}.parquet", index=False)
        print(f"[OK] factor scores persisted for {_d.date()} ({len(result_df)} names)")
    except Exception as _exc:  # noqa: BLE001
        print(f"[WARN] factor score dump failed: {_exc}")

    print(f"\n[OK] Multi-factor weighted predictions calculated for {len(result_df)} stocks")

    return result_df


def main():
    print("\n" + "="*80)
    print("DAILY TRADING SIGNALS - MULTI-FACTOR MODEL")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    artifacts_dir = Path(__file__).parent / "artifacts"
    data_dir = Path(__file__).parent / "data"

    # Load factor ensembles
    print(f"\nLoading factor models...")

    factor_ensembles = {}

    for factor_name in FACTOR_GROUPS.keys():
        model_path = artifacts_dir / f"ensemble_{factor_name}.pkl"

        if not model_path.exists():
            print(f"[WARNING] {factor_name} model not found, skipping")
            continue

        with open(model_path, 'rb') as f:
            factor_ensembles[factor_name] = pickle.load(f)

        print(f"[OK] Loaded {factor_name}: {len(factor_ensembles[factor_name]['feature_cols'])} features")

    if len(factor_ensembles) == 0:
        print("[ERROR] No factor models found!")
        print("Please run: python train_multi_factor_models.py")
        return

    # Load data
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")

    # 2026-08: restrict signal selection to S&P-member equities. ETFs and
    # never-members are baskets/odd names the cross-sectional factors
    # cannot rank; excluding them doubled dev IC (0.0066 -> 0.0129) and
    # holdout IC (0.026 -> 0.051, t=2.65) with no virgin-tier harm --
    # see pit_universe_verdict / commit 09336d5. Same universe as the
    # evaluated pitOFF configuration.
    membership_path = data_dir / "sp500_membership.parquet"
    if membership_path.exists():
        members = set(pd.read_parquet(membership_path)['symbol'].unique())
        n_before = df['symbol'].nunique()
        df = df[df['symbol'].isin(members)]
        print(f"[universe] S&P-member filter: {n_before} -> "
              f"{df['symbol'].nunique()} symbols (ETFs/never-members excluded)")
    else:
        print("[universe][WARN] sp500_membership.parquet missing -- "
              "run data/fetch_sp500_history.py; trading full universe")

    latest_date = df['date'].max()

    print(f"\nUsing data up to: {latest_date.date()}")
    print(f"Stocks available: {len(df[df['date'] == latest_date])}")

    # Calculate historical win rates
    print(f"\nCalculating historical win rates from recent data...")
    win_rates = calculate_win_rates_multi_factor(factor_ensembles, df)

    if win_rates:
        print(f"[OK] Win rates calculated from last {win_rates['sample_size']} samples")
    else:
        print(f"[INFO] Win rates not available")

    # Get multi-factor weighted predictions
    latest_data = get_weighted_predictions_multi_factor(df, factor_ensembles, n_days=5)

    # LONG-TERM TREND FILTER
    print("\n" + "="*80)
    print("LONG-TERM TREND ANALYSIS")
    print("="*80)

    # Calculate long-term metrics for filtering
    print("\nCalculating long-term trend metrics...")

    # Get 7-day, 14-day, 30-day data for trend analysis
    symbols = latest_data['symbol'].unique()
    trend_metrics = []

    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol].sort_values('date')

        if len(symbol_data) < 30:
            continue

        # Get recent prices
        recent_30 = symbol_data.tail(30)
        current_price = recent_30['close'].iloc[-1]

        # Calculate returns
        price_7d_ago = recent_30['close'].iloc[-7] if len(recent_30) >= 7 else current_price
        price_14d_ago = recent_30['close'].iloc[-14] if len(recent_30) >= 14 else current_price
        price_30d_ago = recent_30['close'].iloc[0]

        return_7d = (current_price - price_7d_ago) / price_7d_ago * 100
        return_14d = (current_price - price_14d_ago) / price_14d_ago * 100
        return_30d = (current_price - price_30d_ago) / price_30d_ago * 100

        # Calculate moving averages
        ma_7 = recent_30['close'].tail(7).mean()
        ma_14 = recent_30['close'].tail(14).mean()
        ma_30 = recent_30['close'].mean()

        # Trend strength (MA7 vs MA30)
        trend_strength = (ma_7 - ma_30) / ma_30 * 100

        # Price position relative to MAs
        above_ma7 = current_price > ma_7
        above_ma14 = current_price > ma_14
        above_ma30 = current_price > ma_30

        trend_metrics.append({
            'symbol': symbol,
            'return_7d': return_7d,
            'return_14d': return_14d,
            'return_30d': return_30d,
            'ma_7': ma_7,
            'ma_14': ma_14,
            'ma_30': ma_30,
            'trend_strength': trend_strength,
            'above_ma7': above_ma7,
            'above_ma14': above_ma14,
            'above_ma30': above_ma30
        })

    trend_df = pd.DataFrame(trend_metrics)

    # Merge with latest_data
    latest_data = latest_data.merge(trend_df, on='symbol', how='left')

    # Display trend summary
    print("\nLong-term Trend Summary:")
    print(f"{'Symbol':<8} {'7d%':<8} {'14d%':<8} {'30d%':<8} {'Trend':<8} {'Status':<15}")
    print("-" * 70)

    for _, row in latest_data.iterrows():
        symbol = row['symbol']
        r7 = row.get('return_7d', 0)
        r14 = row.get('return_14d', 0)
        r30 = row.get('return_30d', 0)
        strength = row.get('trend_strength', 0)

        # Determine status
        if r7 < -2 or r14 < -5:
            status = "[!] Declining"
        elif r7 > 2 and r14 > 0:
            status = "[+] Strong Up"
        elif r7 > 0:
            status = "[+] Rising"
        else:
            status = "[-] Mixed"

        print(f"{symbol:<8} {r7:>+6.2f}% {r14:>+6.2f}% {r30:>+6.2f}% {strength:>+6.2f}% {status:<15}")

    # Apply trend filters
    print("\nApplying long-term trend filters...")
    print(f"  7-day return > {FILTER_7D_MIN_RETURN}%")
    print(f"  14-day return > {FILTER_14D_MIN_RETURN}%")
    if FILTER_30D_MIN_RETURN is not None:
        print(f"  30-day return > {FILTER_30D_MIN_RETURN}%")
    print(f"  Trend strength > {FILTER_TREND_STRENGTH_MIN}%")
    if REQUIRE_PRICE_ABOVE_MAs:
        print(f"  Price > MA7 > MA14: Required")

    filtered_before = len(latest_data)

    # Build filter conditions
    filter_conditions = (
        (latest_data['return_7d'] > FILTER_7D_MIN_RETURN) &
        (latest_data['return_14d'] > FILTER_14D_MIN_RETURN) &
        (latest_data['trend_strength'] > FILTER_TREND_STRENGTH_MIN)
    )

    # Optional 30-day filter
    if FILTER_30D_MIN_RETURN is not None:
        filter_conditions = filter_conditions & (latest_data['return_30d'] > FILTER_30D_MIN_RETURN)

    # Optional MA position filter
    if REQUIRE_PRICE_ABOVE_MAs:
        filter_conditions = filter_conditions & latest_data['above_ma7'] & latest_data['above_ma14']

    # Apply filters
    latest_data = latest_data[filter_conditions].copy()

    filtered_after = len(latest_data)
    filtered_count = filtered_before - filtered_after

    if filtered_count > 0:
        print(f"\n[!] FILTERED OUT {filtered_count} stocks due to weak long-term trend")

    print(f"[OK] {filtered_after} stocks passed long-term trend filter")

    # NEWS SENTIMENT FILTER
    print("\n" + "="*80)
    print("NEWS SENTIMENT ANALYSIS")
    print("="*80)

    news_analyzer = NewsSentimentAnalyzer()
    symbols = latest_data['symbol'].tolist()

    print(f"\nFetching news sentiment for {len(symbols)} stocks...")
    news_sentiments = news_analyzer.get_batch_sentiment(symbols, delay=0.5)

    # Add sentiment scores to latest_data
    latest_data['news_sentiment'] = latest_data['symbol'].map(
        lambda s: news_sentiments.get(s, {}).get('sentiment_score', 0.0)
    )
    latest_data['news_count'] = latest_data['symbol'].map(
        lambda s: news_sentiments.get(s, {}).get('news_count', 0)
    )

    # Display sentiment summary
    print("\nNews Sentiment Summary:")
    print(f"{'Symbol':<8} {'Sentiment':<12} {'News Count':<12} {'Status':<15}")
    print("-"*50)

    for symbol in symbols[:10]:  # Show top 10
        sentiment = news_sentiments.get(symbol, {})
        score = sentiment.get('sentiment_score', 0.0)
        count = sentiment.get('news_count', 0)

        if score < -0.5:
            status = "[X] FILTERED OUT"
        elif score < -0.2:
            status = "[!] Negative"
        elif score > 0.5:
            status = "[+] Very Positive"
        elif score > 0.2:
            status = "[+] Positive"
        else:
            status = "[-] Neutral"

        print(f"{symbol:<8} {score:>+6.3f}      {count:<12} {status:<15}")

    # Apply news sentiment boost/penalty to confidence
    print("\nApplying news sentiment adjustment to rankings...")
    print("-" * 80)

    def calculate_sentiment_boost(sentiment_score):
        """Calculate confidence boost based on news sentiment"""
        if sentiment_score >= 0.5:
            return 0.20  # +20% for very positive news
        elif sentiment_score >= 0.2:
            return 0.10  # +10% for positive news
        elif sentiment_score >= -0.2:
            return 0.0   # No change for neutral
        elif sentiment_score >= -0.5:
            return -0.10  # -10% penalty for negative news
        else:
            return -0.50  # -50% penalty for very negative news

    latest_data['sentiment_boost'] = latest_data['news_sentiment'].apply(calculate_sentiment_boost)
    latest_data['original_confidence'] = latest_data['confidence']
    latest_data['adjusted_confidence'] = latest_data['confidence'] * (1 + latest_data['sentiment_boost'])

    # Filter out stocks with extremely negative sentiment (< -0.5)
    negative_threshold = -0.5
    filtered_out = latest_data[latest_data['news_sentiment'] < negative_threshold]['symbol'].tolist()

    if filtered_out:
        print(f"\n[!] FILTERED OUT due to extremely negative news (sentiment < {negative_threshold}):")
        for sym in filtered_out:
            sentiment = news_sentiments.get(sym, {})
            headline = sentiment.get('latest_headline', 'No headline')
            print(f"  - {sym}: {headline[:80]}")

    # Apply news filter
    latest_data = latest_data[latest_data['news_sentiment'] >= negative_threshold].copy()

    print(f"\n[OK] {len(latest_data)} stocks passed news sentiment filter")
    print("\nTop stocks after news sentiment adjustment:")
    print(f"{'Symbol':<8} {'Original':<10} {'Sentiment':<12} {'Boost':<8} {'Adjusted':<10}")
    print("-" * 60)

    top_adjusted = latest_data.nlargest(10, 'adjusted_confidence')
    for _, row in top_adjusted.iterrows():
        symbol = row['symbol']
        orig = row['original_confidence'] * 100
        sent = row['news_sentiment']
        boost = row['sentiment_boost'] * 100
        adj = row['adjusted_confidence'] * 100
        print(f"{symbol:<8} {orig:>7.3f}%   {sent:>+6.3f}       {boost:>+5.1f}%  {adj:>7.3f}%")

    # Select stocks using adjusted confidence (technical + news sentiment)
    config = {
        'n_top': 10,
        'min_confidence': 0.0015,  # 0.15% - Further lowered to select more trading opportunities
        'min_stocks': 3,  # Reduced from 7 to work better with smaller stock pool
        'news_sentiment_threshold': negative_threshold
    }

    # Sort by ADJUSTED confidence (includes news sentiment boost)
    latest_data_sorted = latest_data.sort_values('adjusted_confidence', ascending=False)
    top_n = latest_data_sorted.head(config['n_top'])
    # Use original confidence for threshold to maintain standards
    selected = top_n[top_n['original_confidence'] >= config['min_confidence']]

    n_stocks = len(selected)

    # Display
    print("\n" + "="*80)
    print("TODAY'S SIGNALS (MULTI-FACTOR + 5-DAY WEIGHTED)")
    print("="*80)

    if n_stocks >= config['min_stocks']:
        print(f"\n[TRADE] {n_stocks} stocks recommended\n")

        # Show win rates
        if win_rates:
            print(f"IN-SAMPLE direction check (last {win_rates['sample_size']} rows "
                  f"the models were trained on -- NOT expected live performance;")
            print(f"see evaluation/results/walk_forward_report.json for honest out-of-sample stats):")
            print(f"  Overall: {win_rates['overall_win_rate']*100:.1f}%")
            print(f"  BUY signals: {win_rates['buy_win_rate']*100:.1f}% ({win_rates['buy_total']} BUY predictions)")
            print(f"  SELL signals: {win_rates['sell_win_rate']*100:.1f}% ({win_rates['sell_total']} SELL predictions)")
            print()

        # Position allocation: inverse-volatility scaled confidence.
        # Construction grid (2026-07, portfolio_construction_grid_h20.csv):
        # vol-stabilized weights are consistently more robust out-of-sample
        # (holdout Sharpe 0.74/0.67/0.63 vs 0.43/0.42/0.37 signal-weighted,
        # with shallower drawdowns) -- high-vol names stop dominating risk.
        if 'volatility_20d' in selected.columns:
            _vol = selected['volatility_20d'].clip(lower=0.006)
            _fill = _vol.median() if _vol.notna().any() else 0.02
            selected['sizing_weight'] = (
                selected['adjusted_confidence'] / _vol.fillna(_fill))
        else:
            selected['sizing_weight'] = selected['adjusted_confidence']
        total_conf = selected['sizing_weight'].sum()

        print(f"{'Rank':<6} {'Stock':<8} {'Direction':<10} {'Tech%':<8} {'News':<8} {'Final%':<8} {'Position %':<10}")
        print("-" * 80)

        for i, (_, row) in enumerate(selected.iterrows(), 1):
            symbol = row['symbol']
            pred = row['prediction']
            orig_conf = row['original_confidence']
            sentiment = row['news_sentiment']
            adj_conf = row['adjusted_confidence']
            pos_pct = (row['sizing_weight'] / total_conf) * 100

            direction = "BUY" if pred > 0 else "SELL"

            print(f"{i:<6} {symbol:<8} {direction:<10} {orig_conf*100:>6.2f}%  {sentiment:>+6.3f}  "
                  f"{adj_conf*100:>6.2f}%  {pos_pct:>6.1f}%")

        # Factor breakdown for top 3
        print("\n" + "="*80)
        print("FACTOR CONTRIBUTION ANALYSIS (Top 3 Stocks)")
        print("="*80)

        for i, (_, row) in enumerate(selected.head(3).iterrows(), 1):
            symbol = row['symbol']
            pred = row['prediction']

            print(f"\n{i}. {symbol}: {pred*100:+.2f}% total prediction")
            print("-" * 80)

            for factor_name in factor_ensembles.keys():
                factor_col = f'factor_{factor_name}'
                if factor_col in row:
                    factor_pred = row[factor_col]
                    factor_weight = FACTOR_WEIGHTS[factor_name]
                    contribution = factor_pred * factor_weight

                    print(f"  {factor_name:12}: {factor_pred*100:>+6.2f}% × {factor_weight*100:>3.0f}% = "
                          f"{contribution*100:>+6.2f}% contribution")

        # Allocation
        print("\n" + "-" * 70)
        print("ALLOCATION FOR £100 CAPITAL:")
        print("-" * 70)

        capital = 100

        print(f"\n{'Stock':<8} {'Amount (£)':<15} {'Action':<10}")
        print("-" * 35)

        for _, row in selected.iterrows():
            symbol = row['symbol']
            pred = row['prediction']
            pos_pct = (row['sizing_weight'] / total_conf)
            amount = capital * pos_pct
            action = "BUY" if pred > 0 else "SELL"

            print(f"{symbol:<8} {amount:>8.2f}         {action:<10}")

        print("-" * 35)
        print(f"{'TOTAL':<8} {capital:>8.2f}\n")

    else:
        print(f"\n[SKIP] Insufficient signals ({n_stocks} < {config['min_stocks']})")
        print("Recommendation: Hold cash today\n")

        if win_rates:
            print(f"IN-SAMPLE direction check (last {win_rates['sample_size']} rows, "
                  f"not expected live performance):")
            print(f"  Overall: {win_rates['overall_win_rate']*100:.1f}%")
            print()

    # Risk reminders
    print("\n" + "="*80)
    print("RISK MANAGEMENT CHECKLIST")
    print("="*80)
    print("""
[  ] Set max daily loss at 3% (£3 for £100 capital)
[  ] Set stop-loss at 2-3% per stock
[  ] Plan to sell tomorrow (1-day hold)
[  ] Track all trades in spreadsheet
[  ] Review strategy after 3 consecutive losses
    """)

    # Save signals
    output_file = artifacts_dir / f"signals_multi_factor_{datetime.now().strftime('%Y%m%d')}.json"

    signals = {
        'generated_at': datetime.now().isoformat(),
        'data_date': str(latest_date),
        'method': 'multi-factor with 5-day weighted prediction + news sentiment filter',
        'factor_weights': FACTOR_WEIGHTS,
        'should_trade': n_stocks >= config['min_stocks'],
        'n_stocks': int(n_stocks),
        'stocks': [
            {
                'rank': i + 1,
                'symbol': row['symbol'],
                'prediction': float(row['prediction']),
                'confidence': float(row['original_confidence']),
                'adjusted_confidence': float(row['adjusted_confidence']),
                'position_pct': float((row['sizing_weight'] / total_conf) * 100),
                'news_sentiment': float(row.get('news_sentiment', 0.0)),
                'news_count': int(row.get('news_count', 0)),
                'sentiment_boost': float(row.get('sentiment_boost', 0.0)),
                'factor_contributions': {
                    factor_name: float(row[f'factor_{factor_name}'])
                    for factor_name in factor_ensembles.keys()
                    if f'factor_{factor_name}' in row
                }
            }
            for i, (_, row) in enumerate(selected.iterrows())
        ] if n_stocks >= config['min_stocks'] else [],
        'news_filtered_stocks': filtered_out,
        'config': config
    }

    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"Signals saved to: {output_file}")

    print("\n" + "="*80)
    print("[COMPLETE] Multi-Factor Model with 5-Day Weighted Prediction")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
