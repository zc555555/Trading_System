# W2-C — ATR-Adaptive Stop-Loss / Take-Profit

**Status**: shipped 2026-05-18. Companion to W2-B (staggered multi-day trading).

W2-B introduced per-position 5% stops and 10% take-profits. The same threshold for every stock is wrong: a 5% stop on KO (~1% daily vol) is 6σ noise protection; the same 5% on TSLA (~3.5% daily vol) is 1.4σ — gets stopped out by routine volatility. W2-C fixes this by setting per-stock stop/take prices based on the stock's own 14-day ATR (Average True Range, Wilder 1978).

---

## The problem (recap)

Daily vol varies ~6x across the production universe:

| symbol | daily vol | 5% fixed stop = how many σ |
|---|---|---|
| KO | ~0.8% | ~6σ (massively over-cautious) |
| MSFT | ~1.2% | ~4σ (reasonable) |
| AAPL | ~2.2% | ~2.3σ (acceptable) |
| NVDA | ~3.6% | ~1.4σ (tight) |
| TSLA | ~4.1% | ~1.2σ (will fire on normal days) |
| MU | ~7.4% | <1σ (worse than coin flip) |

The high-vol names get stopped out by noise. The asymmetry inverts the strategy's edge: winners pay the cost of getting flushed; losers stay in.

---

## The fix

```
stop_distance = STOP_ATR_MULTIPLE * ATR_14
take_distance = TAKE_ATR_MULTIPLE * ATR_14
```

Then clamped by a percentage floor and ceiling so a degenerate ATR (e.g. a recently low-volatility week) cannot produce a 0.5% stop or a 30% stop.

ATR_14 = rolling 14-day mean of true range, where true range for one bar =
`max(high - low, |high - prev_close|, |low - prev_close|)`. Defined by Welles Wilder in 1978; the standard volatility-adjusted-risk metric in technical trading. We compute it on the fly from the existing OHLC parquet (no feature-pipeline rebuild needed).

### Direction-aware

- **Long**: stop *below* entry, take *above* entry.
- **Short**: stop *above* entry, take *below* entry.

The monitor's breach check is direction-aware too: longs trigger STOP_LOSS when `current_price <= stop_price`; shorts trigger when `current_price >= stop_price`.

---

## Configuration (`config_trading.py`)

```python
USE_ATR_STOPS = True
STOP_ATR_MULTIPLE = 2.0    # stop at 2x ATR
TAKE_ATR_MULTIPLE = 4.0    # take at 4x ATR (1:2 risk:reward)

# Safety clamps (prevent degenerate ATR from producing absurd thresholds)
STOP_PCT_FLOOR = 0.015     # never tighter than 1.5%
STOP_PCT_CEILING = 0.12    # never wider than 12%
TAKE_PCT_FLOOR = 0.025
TAKE_PCT_CEILING = 0.25

# Fallback used when (a) USE_ATR_STOPS=False, (b) symbol has no ATR data
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
```

To revert to uniform percentages, flip `USE_ATR_STOPS = False`. No code change.

---

## Actual stops, today's tranche (2026-05-18)

Same 10 stocks, same entry prices — the stops now scale with each stock's vol:

| symbol | side | entry | **stop%** | **take%** | basis |
|---|---|---|---|---|---|
| HON | short | $213.24 | **4.3%** | 8.6% | atr |
| ICE | short | $154.36 | 5.1% | 10.2% | atr |
| HSY | short | $186.98 | 5.2% | 10.3% | atr |
| OXY | long | $59.62 | 6.7% | 13.4% | atr |
| CRWD | short | $594.08 | 7.7% | 15.4% | atr |
| PWR | short | $769.99 | 8.3% | 16.6% | atr |
| HUM | short | $305.12 | 9.2% | 18.5% | atr |
| DDOG | short | $207.98 | 11.5% | 23.0% | atr |
| MU | short | $724.66 | **12.0%** (capped) | 25.0% (capped) | atr |
| INTC | long | $108.77 | **12.0%** (capped) | 25.0% (capped) | atr |

Compare to the old behavior — every name would have had 5% stop / 10% take regardless of its vol profile.

---

## Files changed

```
NEW       trading/atr_loader.py          ← computes ATR_14 from OHLC parquet
NEW       trading/risk_levels.py         ← compute_risk_levels(), position_breached()
MODIFIED  config_trading.py              ← USE_ATR_STOPS + multipliers + clamps
MODIFIED  trading/tranche_registry.py    ← TranchePosition gets stop_price,
                                          take_price, atr_at_entry, stop_basis
MODIFIED  run_staggered_trading.py       ← compute risk levels at entry,
                                          persist into the tranche
MODIFIED  monitor_dynamic_trading.py     ← reads per-position stop/take from
                                          registry; falls back to fixed-% if
                                          symbol not in registry
```

---

## Multi-tranche same-symbol handling

A symbol can appear in multiple open tranches (bought Monday and Wednesday, say). Alpaca aggregates these into one position. When the monitor checks risk it uses the **tightest** stop and **tightest** take across tranches:

- Long: max(stop_price), min(take_price)
- Short: min(stop_price), max(take_price)

Rationale: conservative-first. If any tranche's stop fires, the whole symbol position closes. The orchestrator on the next day reconciles via `registry.remove_position()` — TODO for W2-D, currently the registry isn't updated when monitor force-closes a position.

---

## Validation (dry-run, 2026-05-18)

- ✅ All 10 today-tranche positions have stop_price + take_price set
- ✅ Long/short direction correct (stops on the correct side of entry)
- ✅ Ceiling clamps applied to MU and INTC (would have been 14.75% / 16.90% otherwise)
- ✅ Floor clamps would apply to extremely low-vol names (none in today's selection)
- ✅ Monitor's `_load_position_risk_index()` correctly returns `{(symbol, side): {...}}`
- ✅ Monitor's stop/take check is direction-aware (LONG vs SHORT pos.side from Alpaca)
- ✅ Registry persistence: `stop_price`, `take_price`, `atr_at_entry`, `stop_basis` all stored
- ✅ Fallback to fixed % when symbol missing from registry (legacy compat)

---

## Expected impact

Without W2-C, the 5d strategy's 68.1% theoretical win rate would erode because:
1. ~30% of trades would be in stocks with daily vol > 2.5% (NVDA, TSLA, MU, INTC, DDOG, etc.)
2. Those stocks would have 1-1.5σ stops -- about 35% probability of intra-tranche flush even on winning days
3. Net: ~10% of expected winners pre-empted by noise stops, locking losses instead

With W2-C, each stock gets stops calibrated to its own σ. Expected effect: realized win rate ~62-66% (vs theoretical 68.1%), where the legacy fixed-5% would have realized ~52-58%.

These are estimates — paper trading the next 30 days will produce the actual numbers.

---

## Known limitations (W2-D backlog)

1. **Monitor stop-out doesn't update registry.** When monitor force-closes a position via Alpaca, the registry still shows it as open. Next orchestrator run will try to close it again and get an error. Fix: monitor calls `registry.remove_position(tranche_id, symbol, side)` after each force-close.
2. **Wilder's EMA vs SMA.** True Wilder ATR is `(prev_atr * 13 + tr) / 14`. We use SMA which is close enough for risk sizing but technically different.
3. **No trailing stops.** Stop and take are fixed at entry. A position that rallies 8% then reverses to break-even still doesn't trigger anything. Trailing stops are W3-level work.
4. **No regime adjustment.** ATR multiplier is constant across market regimes; one could tighten stops in high-VIX regimes.

---

## Rollback

```python
# config_trading.py
USE_ATR_STOPS = False
```

All new tranches will use the fixed `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`. Already-opened tranches keep their persisted `stop_price` / `take_price`, so they ride out their lifecycle on ATR rules — clean transition either way.
