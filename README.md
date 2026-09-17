# Trading System — an honestly-evaluated multi-factor equity book

A multi-factor US-equity system that trades live (Alpaca paper) as a fully
automated 7-factor book — but whose real subject is **evaluation
methodology**: how to measure a trading signal, and a trading *system*,
without fooling yourself.

> **Start here:** [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)
> (中文版: [docs/RESEARCH_REPORT.zh.md](docs/RESEARCH_REPORT.zh.md))

## The one-paragraph story

This system once reported a backtest Sharpe of 1.89 and a 63.4% win rate.
An audit found a feature reading prices **eleven days into the future**
(~90% of the flagship factor's IC: 20-day IC 0.65 → 0.07 after the fix), a
backtest whose `zip(dict, dict)` bug meant it **never used the models'
predictions**, and an execution path that **bought the stocks the model
wanted to short**. The evaluation stack was rebuilt from scratch — purged
walk-forward, per-date rank IC with Newey-West inference, a four-tier
evidence framework with an untouched holdout, pre-registered adoption
rules, and a hypothesis ledger with family-wise multiple-testing
thresholds. Two further corrections cut the headline again: point-in-time
index membership *inside the portfolio* and survivorship-complete prices
for every departed S&P member took the backtest Sharpe from 0.83 to 0.27.
In September 2026 the evaluation was made to replay the strategy **as it
actually trades** — the production selection policy as one shared module,
and a cash-and-shares ledger with the live book's whole-share sizing,
no-debt/short caps, scheduled exits and (until they were removed on the
evidence) ATR bracket exits. On that ruler the live configuration scores
an all-period Sharpe of roughly **0.4** (dev 0.39, holdout 0.76, max
drawdown −18%); the risk model attributes its +6.3%/yr as market +3.3%,
sector +1.2%, **stock selection +5.4%**, costs −1.0% (avg β ≈ 0.2). The
ledger holds **33 adjudications: 3 adoptions, 18 rejections, 10
measurement corrections and rule changes, 1 watch item, 1 open**. A
sealed LLM agent has proposed **528 factor expressions** under its own
false-discovery-rate family; **none has cleared the holdout gate**. Small
numbers, every one of them defended line by line.

## What makes this repo worth reading

| Artifact | Why it matters |
|---|---|
| [`research/evaluation/`](research/evaluation/) | The trustworthy ruler: purged walk-forward, rank-IC metrics, and `ledger_sim.py`, a cash-and-shares simulator that does what the orchestrator does (whole shares, tranche capital from equity, no-debt/short caps, open-to-open exits, brackets) |
| [`research/strategy/selection.py`](research/strategy/selection.py) | The production selection policy (smoothing, trend filter, top-N, inverse-vol sizing) as one module the nightly script *and* the evaluation call — the two were different strategies until 2026-09 |
| [`research/evaluation/results/hypothesis_ledger.csv`](research/evaluation/results/hypothesis_ledger.csv) | Every hypothesis ever adjudicated, including all failures and the two retractions, with Šidák family-wise thresholds — the multiple-testing record most projects don't keep |
| [`research/evaluation/RULEBOOK.md`](research/evaluation/RULEBOOK.md) + [`research/mining/`](research/mining/) | Track B: a sealed LLM factor miner (fixed DSL, sees development scores only, guard hook denies everything else), Benjamini-Hochberg control over its own family, a factor pool, and process-level oracles (future-data perturbation, implausible strength, membership mask, control expressions) that caught 6/6 injected defects with 0 false alarms |
| [`research/evaluation/experiments/`](research/evaluation/experiments/) | Each experiment as a standalone, re-runnable script with its pre-registered verdict criteria — including `production_replay.py` (the book layer by layer) and `capital_deployment.py` |
| [`research/evaluation/risk_model.py`](research/evaluation/risk_model.py) | Two-layer risk model (market + 11 sectors) with exact-additivity attribution — every construction verdict must carry it |
| [`trading/`](trading/) | Execution safety: an injectable broker with fault injection, filled-only accounting booked per client order id, reconciliation that converges the registry from the broker's own order history and aborts on any failed query, cumulative exposure guards, a persisted circuit breaker, an intraday daily-loss guard, an exact NYSE calendar |
| [`tests/`](tests/) | 244 tests: leakage guards, simulated-broker fault scenarios (failed queries, partial fills, duplicate orders, replacement orders), the selection policy pinned against the arithmetic it replaced, CI on Ubuntu and Windows |
| [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) | The full narrative with every number |

## Architecture

```
                 ┌─────────────────────────────────────────────────────┐
                 │ research/ (offline)                                  │
                 │  data → features → factors → train (rolling 756      │
                 │  sessions, no refit = the evaluated policy)          │
                 │  strategy/selection.py  ← ONE selection policy       │
                 │  evaluation/            ← the only source of         │
                 │    walk-forward + ledger_sim   performance claims     │
                 │  mining/                ← sealed agent factor miner  │
                 └──────────────────┬──────────────────────────────────┘
                                    │ signals_multi_factor_*.json
                                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ Live layer (repo root, fully automated, Task Scheduler)            │
   │  nightly: freshness + calendar gates → staggered 20-day tranches   │
   │           (plain orders; bracket exits removed 2026-09-15 on the   │
   │           replay's evidence) → registry booked by actual fills →   │
   │           reconcile vs broker → attribution → health check         │
   │  intraday: 5-min daily-loss guard (persisted halt, cancel, flatten)│
   │  Sunday:  retrain → staged all-or-nothing model release with a     │
   │           hash manifest → IC-decay monitor → adoption triggers     │
   │  monthly / semi-annual: mining sources refresh, segment rotation   │
   └───────────────────────────────────────────────────────────────────┘
```

**Current book:** 7 factors (volatility 26.0%, 52-week-high 20.0%,
momentum 19.6%, trend 19.2%, alpha 15.1%; market & volume honestly
zero-weighted) · point-in-time S&P members (ETFs excluded — measured, not
assumed) · 5-session score smoothing and a long-term trend filter (both
replayed and adjudicated, both kept) · 20-day staggered tranches · inverse-
volatility sizing · no leverage, single short ≤ 2% and gross short ≤ 25% of
equity (the caps add ~+0.12 dev Sharpe by refusing the weakest short legs)
· 3% daily-loss breaker checked every 5 minutes · calibrated 12–15 bp
round-trip costs (from live fills) · randomized limit-vs-market execution
A/B accumulating evidence · nightly return attribution and health check ·
weekly factor IC-decay monitor with pre-registered WARN/ALERT rules that
*remove* probation adoptions automatically · frozen macro snapshot
(features are a pure function of git) · CI on every push.

## Reproduce the headline results

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

cd research
# honest walk-forward at the production horizon (~15 min)
python evaluation/purged_walk_forward.py --horizon 20 --tag mine
# the book as it trades: production selection through the cash/share ledger
python evaluation/experiments/production_replay.py --horizon 20 --panel surv \
    --data data/stocks_with_time_windows_surv.parquet --engine ledger --no-brackets
# the same, layer by layer (fixed-weight engine / share ledger / + caps / + brackets)
python evaluation/experiments/production_replay.py --horizon 20 --panel surv --data data/stocks_with_time_windows_surv.parquet
python evaluation/experiments/production_replay.py --horizon 20 --panel surv --data data/stocks_with_time_windows_surv.parquet --engine ledger
# run the test suite (leakage guards, evaluation math, simulated-broker scenarios)
cd .. && python -m pytest tests/ -q
```

Factor experiments (each prints its four-tier verdict and ledger entry):

```bash
python evaluation/experiments/factor_pipeline.py --factor high52
python evaluation/experiments/pead_factor.py
python evaluation/experiments/pit_universe_test.py
python evaluation/experiments/capital_deployment.py        # the three deployment levers, all rejected
```

Return attribution through the risk model, and the sealed factor miner:

```bash
python evaluation/attribution_backtest.py --horizon 20 --config surv   # writes docs/img chart
python mining/harness.py ops                                            # operating instructions for the sealed miner
python mining/harness.py --horizon 20 pool status                       # Track P pool and its releases
```

## Data sources

- **Prices:** Sharadar SEP (paid, 2014+, includes every delisted name; the
  survivorship-complete panel is the evaluation standard); yfinance for
  the nightly production panel
- **Index membership:** point-in-time S&P 500 reconstruction from the
  official change log (`data/fetch_sp500_history.py`, 874 tickers / 397
  exit intervals, hard guard against a degraded refresh); a 2,777-ticker
  mid-cap research universe from lagged SEC share counts
- **Fundamentals:** SEC EDGAR XBRL companyfacts with true *filing-date*
  point-in-time discipline, TTM with year-to-date differencing
- **Alternative fields for the miner (each with its own availability rule):**
  GDELT news tone via the BigQuery public dataset, SEC Form 4 insider
  transactions, FINRA short interest, Reg SHO daily short volume, 13F
  institutional ownership, Wikipedia page views
- **Earnings:** yfinance announcement dates + EPS surprises to 2002

## Honest limitations

Paper fills are a lower bound on live costs; the ledger engine has no
borrow fees or cash interest and fills stops at the stop price on daily
bars; verdicts dated before 2026-08-25 were reached on the pre-correction
panel and those before 2026-09-15 on the fixed-weight engine (directions
expected to hold, magnitudes not re-certified); the survivorship-complete
panel is 1.1 GB and rebuilt by `experiments/survivorship_universe.py`
rather than committed; the live book deploys only ~53% of equity because
the short caps bind (the model wants half its book short; the no-debt
principle allows 25%) — every lever to raise deployment was tested and
rejected on drawdown; capacity is small. Full list in
[the report §6](docs/RESEARCH_REPORT.md#6-limitations-stated-plainly).

## Repo map

```
├── research/
│   ├── evaluation/            ← metrics, walk-forward, ledger_sim (cash/share book), experiments, RULEBOOK
│   ├── strategy/selection.py  ← the production selection policy, shared with the evaluation
│   ├── mining/                ← sealed agent factor miner: DSL, harness, oracles, pool, memory
│   ├── data/                  ← fetchers (Sharadar, membership, EDGAR, GDELT, Form 4, FINRA, 13F, Wikipedia)
│   ├── features/ factors/     ← feature engineering, factor definitions, mined-factor registry, model release
│   └── train_multi_factor_models.py
├── run_auto_trading.py        ← nightly pipeline (unattended-safe, per-step timings)
├── run_staggered_trading.py   ← tranche orchestrator (gates, filled-only booking, reconcile, halt)
├── run_health_check.py        ← nightly outcome / data / signals / models / book / halt check
├── trading/                   ← broker + fake broker, execution, halt, intraday guard, registry, NYSE calendar
├── scripts/                   ← Task Scheduler entry points and registrations
├── tests/                     ← leakage guards, evaluation math, simulated-broker scenarios
└── docs/RESEARCH_REPORT.md    ← the full story
```

## Disclaimer

Educational and research purposes only; not financial advice. The system
runs in paper-trading mode.
