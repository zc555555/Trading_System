# Trading System — an honestly-evaluated multi-factor equity book

A multi-factor US-equity system that trades live (Alpaca paper) as a fully
automated 7-factor book — but whose real subject is **evaluation
methodology**: how to measure a trading signal without fooling yourself.

> **Start here:** [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)
> (中文版: [docs/RESEARCH_REPORT.zh.md](docs/RESEARCH_REPORT.zh.md))

## The one-paragraph story

This system once reported a backtest Sharpe of 1.89 and a 63.4% win rate.
An audit found a feature reading prices **eleven days into the future**
(~90% of the flagship factor's IC), a backtest whose `zip(dict, dict)` bug
meant it **never used the models' predictions**, and an execution path that
**bought the stocks the model wanted to short**. The evaluation stack was
rebuilt from scratch — purged walk-forward, per-date rank IC with
Newey-West inference, a four-tier evidence framework with an untouched
holdout, pre-registered adoption rules, and a hypothesis ledger with
family-wise multiple-testing thresholds. The rebuilt ruler has since
adjudicated **20 hypotheses (3 adopted, 16 rejected, 1 watch-listed)**,
twice overturning findings that looked significant (t > 2) on partial
evidence. The honest expectation of the live configuration is a Sharpe of
roughly **0.8, with a measured beta component** (a two-layer risk model
attributes the +13.7%/yr headline as market +8.7%, sector +2.9%, stock
selection +4.5%, costs −1.9%; avg β ≈ 0.59) — a small number that can
be defended line by line.

## What makes this repo worth reading

| Artifact | Why it matters |
|---|---|
| [`research/evaluation/`](research/evaluation/) | The trustworthy ruler: purged walk-forward, rank-IC metrics, cost-aware next-open simulation — one audited path for every performance claim |
| [`research/evaluation/results/hypothesis_ledger.csv`](research/evaluation/results/hypothesis_ledger.csv) | Every hypothesis ever adjudicated, including all failures, with Šidák family-wise thresholds — the multiple-testing record most projects don't keep |
| [`research/evaluation/experiments/`](research/evaluation/experiments/) | Each experiment as a standalone, re-runnable script with its pre-registered verdict criteria |
| [`research/evaluation/risk_model.py`](research/evaluation/risk_model.py) | Two-layer risk model (market + 11 sectors) with exact-additivity attribution — the beta caveat as a daily number, backtest and live |
| [`tests/`](tests/) | Leakage guards that structurally forbid negative feature shifts and backward-fills — the audited bug classes cannot silently return |
| [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) | The full narrative with every number |

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │ research/ (offline)                          │
                 │  data → features → factors → train           │
                 │  evaluation/  ← the only source of           │
                 │                 performance claims           │
                 └──────────────────┬──────────────────────────┘
                                    │ signals_multi_factor_*.json
                                    ▼
   ┌───────────────────────────────────────────────────────────┐
   │ Live layer (repo root, fully automated)                    │
   │  nightly: signals → staggered tranches w/ broker-side      │
   │           GTC brackets → registry                          │
   │  intraday: 5-min monitor (kill switch, reconciliation,     │
   │            limit-order timeout conversion)                 │
   │  Sunday:  full retrain + slippage/A-B reconciliation       │
   └───────────────────────────────────────────────────────────┘
```

**Current book:** 7 factors (volatility 28.8%, 52-week-high 22.9%, trend
18.8%, alpha 16.9%, momentum 12.6%; market & volume honestly zero-weighted)
· 273 point-in-time S&P members (ETFs excluded — measured, not assumed)
· 20-day staggered tranches · inverse-volatility sizing · calibrated
12–15 bp round-trip costs (from live fills) · randomized limit-vs-market
execution A/B accumulating evidence · nightly return attribution of the
live book (market / sector / selection).

## Reproduce the headline results

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

cd research
# honest walk-forward at the production horizon (~15 min)
python evaluation/purged_walk_forward.py --horizon 20 --tag mine
# cost-aware portfolio simulation from the panel
python evaluation/simulate_portfolio.py --horizon 20
# run the leakage-guard and evaluation test suite
cd .. && python -m pytest tests/ -q
```

Factor experiments (each prints its four-tier verdict and ledger entry):

```bash
python evaluation/experiments/factor_pipeline.py --factor high52
python evaluation/experiments/pead_factor.py
python evaluation/experiments/pit_universe_test.py
```

Return attribution of the headline book through the risk model:

```bash
python evaluation/attribution_backtest.py --horizon 20   # writes docs/img chart
```

## Data sources (all free)

- **Prices:** yfinance OHLCV, 2014+ (survivorship residual documented)
- **Index membership:** point-in-time S&P 500 reconstruction from the
  official change log (`data/fetch_sp500_history.py`)
- **News tone:** GDELT via the BigQuery public dataset (no API throttle)
- **Earnings:** yfinance announcement dates + EPS surprises to 2002
- **Fundamentals:** SEC EDGAR XBRL companyfacts with true *filing-date*
  point-in-time discipline (`data/fetch_fundamentals.py`)

## Honest limitations

Paper fills are a lower bound on live costs; the book carries a measured
beta component (avg β ≈ 0.59, ~2/3 of gross return systematic; the
attribution chart in the report shows the split); delisted names' prices
are unavailable (only the inclusion half of survivorship bias is fixed);
capacity is small. Full list in
[the report §6](docs/RESEARCH_REPORT.md#6-limitations-stated-plainly).

## Repo map

```
├── research/
│   ├── evaluation/            ← metrics, walk-forward, simulator, experiments
│   ├── data/                  ← fetchers (prices, membership, GDELT, EDGAR)
│   ├── features/ factors/     ← feature engineering, factor definitions
│   └── train_multi_factor_models.py
├── run_auto_trading.py        ← nightly pipeline (unattended-safe)
├── run_staggered_trading.py   ← tranche orchestrator (brackets, reconcile)
├── monitor_dynamic_trading.py ← 5-min risk monitor
├── trading/                   ← tranche registry, risk levels, calendar
├── tests/                     ← leakage guards + evaluation unit tests
└── docs/RESEARCH_REPORT.md    ← the full story
```

## Disclaimer

Educational and research purposes only; not financial advice. The system
runs in paper-trading mode.
