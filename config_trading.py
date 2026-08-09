"""Trading-layer configuration (W2-B, 2026-05).

Separated from research/config.yaml because trading params change
independently of model params.

Edit this file to switch strategies, change holding periods, or tune risk.
"""

# ----------------------------------------------------------------------
# Strategy selection
# ----------------------------------------------------------------------
# "legacy"   : 1-day hold, daily flatten at EOD (the original W1 system).
# "staggered": Multi-day hold with overlapping tranches (W2-B). Each day
#              deploys 1/hold_days of capital; oldest tranche closes when
#              its scheduled_close_date arrives.
STRATEGY = "staggered"

# ----------------------------------------------------------------------
# Staggered strategy parameters (ignored when STRATEGY = "legacy")
# ----------------------------------------------------------------------
# Number of trading days each tranche is held before forced close.
# 5 = weekly turnover, 20 = monthly.
# 2026-07 (P2/D): switched 5 -> 20. The honest walk-forward showed the
# 20-day-hold configuration is the only one near breakeven after costs
# (round-trip cost amortized to ~1.5bp/day); 1-day and 5-day holds lose
# to cost drag. See evaluation/results/simulation_report_h20.json.
HOLD_DAYS = 20

# 100 / HOLD_DAYS = pct of equity deployed per new tranche.
# Leave None to auto-derive from HOLD_DAYS.
CAPITAL_PER_TRANCHE_PCT = None

# Hard ceiling per individual stock per tranche (post-allocation cap).
# Stops one signal from eating 20% of a tranche if many signals fail.
PER_STOCK_MAX_PCT = 30

# Allow short positions for negative-prediction signals?
# Requires Alpaca short-selling approval (free on paper accounts).
ALLOW_SHORTS = True

# If True, drop SELL signals entirely instead of shorting them.
# Use when ALLOW_SHORTS is False or short approval is unavailable.
LONG_ONLY_FILTER = False

# ----------------------------------------------------------------------
# Risk parameters (apply to BOTH strategies)
# ----------------------------------------------------------------------
# Stop-loss model:
#   USE_ATR_STOPS = True  -> per-position stop/take prices computed at entry
#       from the stock's own ATR_14 (Average True Range). High-vol names like
#       NVDA get wider stops, low-vol names like KO get tighter ones.
#       This is the standard private-fund approach.
#   USE_ATR_STOPS = False -> the legacy uniform percentage below applies to
#       every position regardless of stock.
USE_ATR_STOPS = True

# ATR multiples (used when USE_ATR_STOPS = True).
#   stop_distance = STOP_ATR_MULTIPLE * ATR_14
#   take_distance = TAKE_ATR_MULTIPLE * ATR_14
# 2026-07: widened 2.0/4.0 -> 3.0/6.0 for the 20-day hold. A 20-day
# position sees ~2x the price noise of a 5-day one; the validated
# simulation held positions to schedule with NO stops, so stops here are
# disaster protection, not an active exit strategy. Tight stops would
# make live behavior diverge from the evaluated configuration.
STOP_ATR_MULTIPLE = 3.0
TAKE_ATR_MULTIPLE = 6.0

# Fallback uniform % when ATR data is missing (e.g. brand-new ticker).
# Also the legacy values used when USE_ATR_STOPS = False.
# Widened from 2.5% (1-day hold) -> 5% for multi-day holds because 5-day
# positions have ~2x daily vol of single-day positions.
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# Sanity-check bounds for ATR-derived stops (prevents a degenerate ATR
# reading from producing 50% stops or 0.1% stops).
STOP_PCT_FLOOR = 0.015    # never tighter than 1.5%
STOP_PCT_CEILING = 0.12   # never wider than 12%
TAKE_PCT_FLOOR = 0.025
TAKE_PCT_CEILING = 0.25

# Account-wide intraday kill switch.
MAX_DAILY_LOSS_PCT = 0.03

# ----------------------------------------------------------------------
# Execution A/B test (2026-08, merged-B item 1)
# ----------------------------------------------------------------------
# When True, each new entry order is deterministically assigned by
# hash(symbol+date) to:
#   arm A ("market"): market bracket (the incumbent execution)
#   arm B ("limit") : limit bracket at the arrival price; the monitor
#                     converts non-fills to market LIMIT_TIMEOUT_MIN
#                     minutes into the regular session.
# Realized slippage per arm is compared by the recurring
# slippage_calibration report. Verdict needs ~1-2 months of fills.
EXECUTION_AB_TEST = True
LIMIT_TIMEOUT_MIN = 30

# ----------------------------------------------------------------------
# Monitor behavior
# ----------------------------------------------------------------------
# If True, monitor closes ALL positions at EOD_CLOSE_TIME ET. This is the
# correct behavior for STRATEGY = "legacy" but MUST be False for STRATEGY =
# "staggered" -- otherwise tranches die after 1 day defeating the whole point.
EOD_FLATTEN = (STRATEGY == "legacy")

EOD_CLOSE_TIME_ET = "16:00"  # Only used if EOD_FLATTEN is True.

# ----------------------------------------------------------------------
# File paths
# ----------------------------------------------------------------------
TRANCHE_REGISTRY_PATH = "trading_logs/position_tranches.json"


def derived_capital_per_tranche_pct() -> float:
    """Resolve CAPITAL_PER_TRANCHE_PCT (uses HOLD_DAYS if not set explicitly)."""
    if CAPITAL_PER_TRANCHE_PCT is not None:
        return float(CAPITAL_PER_TRANCHE_PCT)
    return 100.0 / HOLD_DAYS


def summary() -> str:
    if USE_ATR_STOPS:
        stop_desc = f"stop={STOP_ATR_MULTIPLE}xATR"
        take_desc = f"take={TAKE_ATR_MULTIPLE}xATR"
    else:
        stop_desc = f"stop={STOP_LOSS_PCT*100:.1f}%"
        take_desc = f"take={TAKE_PROFIT_PCT*100:.1f}%"
    return (
        f"STRATEGY={STRATEGY}, hold_days={HOLD_DAYS}, "
        f"capital_per_tranche={derived_capital_per_tranche_pct():.1f}%, "
        f"shorts={'on' if ALLOW_SHORTS else 'off'}, "
        f"{stop_desc}, {take_desc}, "
        f"eod_flatten={EOD_FLATTEN}"
    )


if __name__ == "__main__":
    print(summary())
