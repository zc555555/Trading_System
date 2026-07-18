# W1 — Universe Expansion (60 → ~200 with liquidity filter)

**Status**: shipped 2026-05-17. Backwards-compatible with explicit `data.symbols` rollback.

This is the first concrete step of the 16-week upgrade plan. Goal: stop wasting 95% of model capacity by enlarging the training & trading universe, while keeping illiquid names from polluting signals.

---

## What changed

| Layer | Before | After |
|---|---|---|
| Stock list source | 60 tickers hardcoded in `research/config.yaml` | Named universe resolved at runtime (`sp100` / `sp500_safe` / `sp500` / `all`) |
| Default universe | 60 ad-hoc names | **`sp500_safe`** → ~200 high-conviction large/mid-caps (offline-safe) + benchmark/sector ETFs |
| Liquidity gating | none | Drops symbols with 20d ADV < $50M, price < $5, or history < 252d |
| Daily pipeline | 3 steps | 4 steps — liquidity filter inserted between fetch and feature build |
| Vestigial config field | `data.trading_symbols` (unused by any code) | Removed |

---

## New files

```
research/
├── config_loader.py                       ← Single entry point for reading config.yaml
└── data/
    ├── universe.py                        ← Named universes + Wikipedia S&P 500 fetcher
    └── apply_liquidity_filter.py          ← Post-fetch parquet pruning
```

## Modified files

- `research/config.yaml` — switched to named universe + liquidity-filter parameters
- `research/data/fetch_ohlcv.py` — resolves universe via config; writes `fetch_manifest.json`
- `research/data/fetch_news_sentiment.py` — uses `config_loader.load_config()`
- `research/data/fetch_gdelt_news.py` — uses `config_loader.load_config()`
- `research/data/fetch_option_iv.py` — uses `config_loader.load_config()`
- `run_auto_trading.py` — inserts `apply_liquidity_filter.py` after fetch

---

## How to run W1 for the first time

From the project root:

```bash
# Activate venv first (Windows: venv\Scripts\activate)

# Step 1 — wipe old parquet so the new universe gets a clean rebuild
del research\data\stocks.parquet     # Windows
# rm research/data/stocks.parquet    # macOS/Linux

# Step 2 — fetch the new universe (~200 names, ~30-60 min on free yfinance)
cd research
python data/fetch_ohlcv.py

# Step 3 — apply liquidity filter
python data/apply_liquidity_filter.py

# Step 4 — rebuild features
python prepare_prediction_data.py

# Step 5 — retrain all factor models on the larger universe (from project root)
cd ..
python run_retrain_models.py
```

Then run the normal daily flow as usual:

```bash
python run_auto_trading.py
```

---

## Switching universe size

Edit `research/config.yaml`:

```yaml
data:
  universe: "sp500_safe"        # ~200, offline-safe (default)
  # universe: "sp100"           # ~100, mega-caps only — fastest training
  # universe: "sp500"           # ~500, live Wikipedia fetch
  # universe: "all"             # sp500 + benchmark ETFs

  include_benchmark_etfs: true   # adds SPY/QQQ/IWM/sector ETFs as tradables
```

After changing, re-run the four steps above (delete parquet → fetch → filter → prepare).

---

## Rollback

If anything breaks and you need the old 60-symbol behavior, edit `config.yaml`:

```yaml
data:
  symbols:                       # explicit list overrides `universe`
    - AAPL
    - MSFT
    # ...
```

The explicit `symbols` list always wins over `universe` in `config_loader.load_config()`. There is also a one-shot backup of the pre-filter parquet at:

```
research/data/stocks.parquet.prefilter.bak
```

---

## Tuning the liquidity filter

Defaults in `config.yaml`:

```yaml
liquidity_filter:
  enabled: true
  min_dollar_volume_20d: 50000000   # $50M ADV — Alpaca paper can handle smaller, but this avoids slippage surprises in live trading
  min_price: 5.0                    # drop sub-$5 names (often manipulated)
  min_history_days: 252             # need ≥ 1y for 250-day features
```

To preview what gets dropped without writing anything:

```bash
cd research
python data/apply_liquidity_filter.py --dry-run
```

Per-symbol diagnostics land in `research/data/liquidity_report.json` after every run (dry-run included).

---

## What this DID NOT change

- Models. The 6 factor ensembles (`research/artifacts/ensemble_*.pkl`) are not retrained automatically — you must run `run_retrain_models.py` after the new parquet exists. Until then, signals will only generate for the intersection of the new universe and the old feature manifest.
- Signal selection logic. `get_daily_signals_multi_factor.py` still takes `n_top=10` and `min_confidence=0.0015`. With a bigger pool, you'll naturally get more candidates clearing the bar — tune later in W2 if too many fire.
- Trading code (`alpaca_trader.py`, `monitor_dynamic_trading.py`). Untouched.

---

## Sanity check after the first run

```bash
python research/data/universe.py
# → prints universe sizes; sp100 ≈ 100, sp500_safe ≈ 200, sp500 ≈ 503

cat research/data/fetch_manifest.json
# → which tickers got data, which failed yfinance

cat research/data/liquidity_report.json | jq '.kept_count, .dropped_count'
# → expect kept ≈ 150-200, dropped ≈ 0-50 (mostly history failures from recent IPOs)
```

If `kept_count` < 100, the filter is too strict — lower `min_dollar_volume_20d` to `20_000_000`.

---

## Next: W2

Data infrastructure refactor — partitioned parquet, DuckDB query layer, incremental `daily_ingest.py`. Plan in the project root `计划` brief.
