# Honest Alpha: Auditing, Rebuilding, and Stress-Testing a Multi-Factor Equity System

*Research report — August 2026. All numbers in this report are reproducible from the committed evaluation code (`research/evaluation/`) and are recorded with the exact commits that produced them.*

---

## Executive summary

This project began as a multi-factor US-equity paper-trading system that reported a backtest Sharpe of 1.89 and a 63.4% win rate. **Both numbers were false.** A line-by-line audit found a single feature reading prices eleven days into the future, a backtest that had never actually used its models' predictions, and an execution path that bought the stocks the model wanted to short.

Rather than patch and move on, I rebuilt the entire evaluation methodology — purged walk-forward validation, per-date cross-sectional rank IC with Newey-West inference, a four-tier evidence framework with an untouched holdout, pre-registered adoption rules, and a hypothesis ledger that applies family-wise multiple-testing corrections to every claim the project has ever made.

The rebuilt ruler then adjudicated **21 hypotheses: 3 adoptions, 17 rejections, 1 watch-list item**. Two of the rejections overturned findings that had looked statistically significant (t > 2) on partial evidence — caught by deliberately mining *never-before-evaluated* data rather than waiting for the future to arrive. The system now trades live (paper) as a fully automated 7-factor book whose honest expectation is a Sharpe of roughly 0.8 with a measured beta component (a risk model attributes the return daily: market, sector, stock-selection — §3.6) — a number I can defend line by line, which the fake 1.89 never was.

The thesis of this report: **for a research career, the ability to produce a trustworthy zero matters more than the ability to produce an untrustworthy two.**

---

## 1. The audit: three ways to fool yourself

### 1.1 A feature that read the future

The momentum factor included a Detrended Price Oscillator implemented as:

```python
dpo_values = group[column].shift(-shift_period) - sma_values   # shift(-11)
```

`shift(-11)` injects the closing price from **eleven trading days in the future** into a live feature (the textbook DPO uses a *positive* shift). Retraining on identical data with only this fix applied:

| Factor | Horizon | IC with leak | IC after fix |
|---|---|---|---|
| momentum | 1d | +0.2586 | **+0.0199** |
| momentum | 5d | +0.5813 | **+0.0145** |
| momentum | 20d | +0.6452 | **+0.0666** |
| all other factors | all | — | *bit-for-bit unchanged* |

The five factors that did not contain the leaky feature reproduced to the last decimal — a natural control group proving the pipeline is deterministic and the collapse is entirely attributable to the fix. **Roughly 90% of the flagship factor's reported edge was one look-ahead bug.**

### 1.2 A backtest that never consulted the model

```python
for model, weight in zip(models, weights):   # models is a DICT
    pred = model.predict(X)                  # 'xgb'.predict -> AttributeError
    ...
except Exception:
    continue                                 # silently swallowed
```

Iterating two dicts with `zip` yields their *keys*; the string `'xgb'` has no `.predict`; the bare `except` swallowed the error every day. Every prediction stayed zero and the backtest traded an arbitrary fixed basket for its entire history. The Sharpe-1.89 equity curve measured nothing.

### 1.3 An execution path that inverted the signal

The order loop placed `'buy'` unconditionally — SELL signals (about half of all signals) were *bought*, with a dead position-size cap. Independently, the advertised 63.4% win rate was computed on the most recent 500 rows *the models had been trained on*.

**Lesson encoded into the project:** every performance number must come from one audited evaluation path; `tests/` now contains leakage guards that structurally forbid negative feature shifts and backward-fills, so these bug classes cannot silently return.

---

## 2. The rebuilt ruler

### 2.1 Purged walk-forward

Rolling 3-year train / 1-quarter test folds. Between every train window and its test window, `horizon + 5` trading days are purged so no training label overlaps evaluation data; the same purge splits each fold's inner validation, which alone determines model and factor weights. Non-overlapping test windows concatenate into a fully out-of-sample prediction panel (~2,100 trading days, 2017–2026). An optional partial tail fold lets the newest data enter evaluation as its labels mature.

### 2.2 Per-date rank IC with honest inference

The old pipeline scored factors with pooled Pearson correlation over all validation rows — a statistic inflated by time-series co-movement and, for multi-day labels, by overlapping-window autocorrelation (the mechanism behind the leak-era "IC rises with horizon" illusion). The honest question for a cross-sectional book is: *on each day, how well do predictions rank that day's returns?* We compute per-date Spearman IC and report Newey-West t-statistics with `h−1` lags, since consecutive daily ICs at an h-day horizon share overlapping return windows. The naive t overstates significance by roughly √h.

### 2.3 Four evidence tiers

Every experiment reports the same four segments:

| Tier | Period | Role |
|---|---|---|
| `virgin_early` | 2016 – 2021 | data no experiment had ever evaluated (obtained by *extending history backward*) |
| `seen_dev` | 2022 – 2025 H1 | development data — the only segment adoption decisions may optimize |
| `holdout` | 2025-07 – 2026-05 | looked at only for confirmation, never selection |
| `fresh` | after 2026-05 | genuinely new data that did not exist when hypotheses were formed |

The backward extension matters: when a candidate factor looked adoptable on holdout evidence, we did not wait months for new data — we mined the *past* for data the factor had never seen. This adjudicated two would-be adoptions in an afternoon (§4).

### 2.4 The hypothesis ledger

`evaluation/results/hypothesis_ledger.csv` records **every hypothesis this project has ever adjudicated** — including all failures. Each new verdict prints the family size N and the Šidák-corrected t threshold for a 5% family-wise error rate (currently |t| ≥ 2.97 at N = 20). Testing 16 worthless ideas gives a 56% chance that at least one shows t > 2 by luck; the ledger is what makes that arithmetic impossible to forget. It caught a t = 7.5 mirage (a seasonality factor whose "significance" was one hot calendar month spanning ~2 independent observations) and forced honest labeling of a t = 2.88 near-miss (§5).

---

## 3. What the honest ruler measured

### 3.1 Baseline: the signal was worth nothing at daily frequency

At the original 1-day horizon, the blended out-of-sample rank IC is **+0.0025 (t = 0.57)** over 1,071 days — statistically zero. Simulated with the then-assumed 30 bp round-trip cost and next-open execution, the 1-day book loses at a Sharpe of −4.05: pure cost drag on a no-edge signal ((1−0.003)²⁵² ≈ −53%/yr). The reported 63.4% win rate is unreachable from these numbers; it was an in-sample artifact.

### 3.2 Horizon: signal quality and cost amortization both favor longer holds

| Label horizon | Blended rank IC (NW t) | Matched-hold simulation |
|---|---|---|
| 1 day | +0.0025 (0.57) | Sharpe −4.05 |
| 5 days | +0.0094 (1.48) | Sharpe −0.85 |
| 20 days | +0.0071 (0.52) | **Sharpe +0.04 (breakeven)** |

First honestly significant components: **trend@5d (+0.0108, t = 2.27)** and **volatility@5d (+0.0177, t = 2.06)**, volatility being positive in every tier at every horizon. The leak-era conclusion "longer horizons are better" was directionally right for the wrong reasons: honest ICs are ~40× smaller than the leaked ones, and the horizon's true benefit is cost amortization (30 bp over 20 days ≈ 1.5 bp/day), not IC growth.

### 3.3 Execution: measured, not assumed

Three weeks of live paper fills (27+ orders) calibrated the cost model: intraday market orders paid a *median +3.8 bp* versus the submission quote; overnight-queued orders were *unbiased* versus the next official open (mean −2.5 bp, median +6.6 bp, high per-name variance from paper's first-quote fills). With zero commission, the realistic round trip is **12–15 bp, half the modeled 30 bp** — this measurement alone moved the 20-day book's honest expectation from ~0 to dev Sharpe ~0.7. Caveat recorded with the data: paper fills simulate NBBO without market impact and are a lower bound on live costs. A randomized **limit-vs-market execution A/B** (deterministic per-order assignment, monitor-driven timeout-to-market) is accumulating live evidence.

### 3.4 Portfolio construction: one upgrade, one honest discovery, one rejected optimizer

A 12-configuration grid (width × weighting × neutrality × cost) found exactly one robust improvement: **inverse-volatility position sizing** (holdout Sharpe 0.74/0.67/0.63 across widths versus 0.43/0.42/0.37 signal-weighted, with shallower drawdowns) — adopted into production. The grid also surfaced an uncomfortable truth: dollar-neutralizing hurts every configuration, meaning the book's returns carry a **real beta component**; the pure alpha is thinner than the headline. This is reported, not hidden — and §3.6 puts a number on it.

A cvxpy mean-variance optimizer (Ledoit-Wolf covariance, 10% vol target, |β| ≤ 0.5, transaction costs in the objective) **lost to the heuristic** (dev Sharpe 0.44 vs 0.75): horizon-amortized decision costs produced 21.7%/day turnover against a 20-day signal (~8%/yr cost drag), and the beta cap removed drift the incumbent keeps. Two implementation lessons are documented — the myopic-objective zero-trade equilibrium, and a DCP violation in a relative beta bound. *A carefully calibrated heuristic beats an uncalibrated optimizer* is itself a result.

### 3.5 The universe was quietly lying about the past

Point-in-time S&P 500 membership (reconstructed from the index change log; 500–504 members at every checkpoint) measured the membership look-ahead bias directly: **pre-2022 IC was ~85% inclusion-runup artifact** (virgin-tier IC 0.0092 → 0.0014 under PIT filtering), while 2022+ results — where every adoption decision actually lived — were untouched, and every prior rejection survives *a fortiori*. A bonus finding was adopted into production: excluding ETFs/never-members from signal selection doubles dev IC (0.0066 → 0.0129) and lifts holdout IC to 0.051 (t = 2.65) — baskets dilute a rankable cross-section. Documented residual: delisted members' price histories remain unavailable with free data; only the inclusion half of survivorship bias is fixed.

### 3.6 Risk model and daily attribution: the beta, named

The "carries beta" caveat deserved a number, so the book got a **two-layer linear risk model** (rolling 252-day SPY beta, estimated ex-ante — the beta attributing day *t* uses data through *t−1* — plus 11 equal-weight sector factors built from market residuals) and a daily attribution that decomposes every session's return into market, sector, stock-selection, cost, and execution-timing components. The decomposition is exact by construction (market + sector + selection ≡ Σ w·r, pinned by unit tests), and the residual is reported separately rather than flattering the selection line.

The verdict on the full-history headline book (h=20, members universe, inverse-vol, calibrated costs — annualized +13.7%):

| Component | dev | holdout | all | IR (all) |
|---|---|---|---|---|
| Market (avg β ≈ 0.59) | +8.5% | +10.4% | **+8.7%** | 0.68 |
| Sector tilts | +2.6% | +4.4% | **+2.9%** | 0.58 |
| **Stock selection** | +3.8% | +9.4% | **+4.5%** | 0.55 |
| Costs | −1.9% | −1.8% | −1.9% | — |
| Execution residual | −0.3% | −1.7% | −0.5% | — |

![Cumulative return attribution](img/attribution_prod_h20.png)

So roughly **two-thirds of the gross return (and 66% of the variance) is systematic** — the honest reading of the Sharpe ≈ 0.8 headline — but the stock-selection component is *positive in both evaluation segments* with an information ratio of ~0.5: small, real, and now measured rather than asserted. The same engine attributes the live paper book every night (`log_attribution.py` → `trading_logs/attribution_history.csv`), accumulating the live answer to the same question.

The risk model's first downstream consumer was pre-registered and **rejected**: a volatility-target overlay (12% target, factor-covariance forecast, scale-*down*-only with a hard 1.0 cap applied to each day's new tranche) delivered exactly what it promised on risk — dev vol 17.0% → 14.5%, max drawdown −23.0% → −21.3% — but charged more Sharpe than the pre-registered band allowed (dev 0.75 → 0.63 against a 0.05 tolerance; holdout confirmed the damage, 1.39 → 1.26). The mechanism is recorded with the verdict: a long-biased book earns much of its return in high-volatility recoveries, and a vol-timing rule sells precisely those days. The overlay stays out of production; the forecaster stays as a monitoring instrument.

---

## 4. The factor program: 21 hypotheses, 3 survivors

The complete ledger, most instructive cases first:

**PEAD (earnings drift) — rejected by the deciding vote it pre-registered.** Built from 25.8k earnings announcements (surprise, reaction, decay features; announcement info usable only from the second session after the date). On July 30 it showed *two independently significant tiers* (virgin +0.0168, t = 2.05; holdout +0.0430, t = 2.24) — adoption-worthy at first glance. Under the pre-registered rule "the fresh tier casts the deciding vote," 40 more days of virgin data returned **−0.0239 (t = −2.77)**: a sign-oscillating factor, not a stable edge. The snapshot that looked adoptable would have put a decaying factor into production.

**vol_ext (8 range/idiosyncratic volatility features) — rejected by mined history.** Dev-neutral, holdout-strong (t = 2.5). Extending the dataset from 2018 back to 2014 created four years of folds the factor had never seen: its virgin IC was *negative* (−0.0077). The entire edge lived in one 10-month window.

**high52 (52-week-high anchoring, George & Hwang 2004) — conditionally adopted.** The strongest profile of any candidate: standalone IC positive in dev (+0.020), holdout (+0.032, t = 2.62) and fresh (+0.066), flat in virgin; blend dev IC **0.0129 → 0.0314**, the largest dev gain of any tested change. Its holdout t = 2.88 sits *below* the Šidák family bar of 2.93 — so adoption is explicitly conditional, with a pre-registered removal trigger (negative fresh tier at the November 2026 review). In the retrained 7-factor production blend it independently earned a 22.9% weight, second only to volatility.

**News tone (GDELT, 262 symbols, 2017+) — validated at 5 days, correctly not deployed at 20.** Passed all three pre-registered conditions at h=5 (sign-positive in three tiers including both virgin ones) but its IC is flat zero at the 20-day production horizon — news information decays. It sits on the shelf, validated, for any future 5-day book.

**Label engineering — three rejections with one architectural lesson.** De-marketed labels were redundant (per-date z-scoring already neutralizes the market component at prediction time); trailing-beta labels added estimation noise; vol-scaled labels double-counted an adjustment the portfolio layer already applies. *Risk and market neutralization should each live in exactly one layer.*

**Fundamentals (SEC EDGAR XBRL) — infrastructure permanent, factors mortal.** A point-in-time pipeline where every value carries its first *filing* date (the day the market learned it), TTM aggregates require all four quarters filed, Q4 flows derived as FY − 3Q, YTD cashflow cumulatives differenced into quarters, tag-switch histories unioned. Verdicts: value dead at 20 days (as pre-discounted), profitability flat everywhere, investment vetoed by a significantly negative fresh tier (t = −3.28) — and **accruals (Sloan 1996) the closest miss in the project** (fresh t = 2.86 vs bar 2.97, no negative tier): watch-listed for the November review. Meta-result: free fundamentals add nothing to a price-based blend at 20 days under these bars.

Also rejected: Amihud illiquidity and short-term reversal (sign oscillators), calendar seasonality (the t = 7.5 mirage), and the leak-era conclusions themselves.

---

## 5. The live system

Fully automated since 2026-07-21: nightly signal generation and staggered order placement (Mon–Fri 21:00 UK), 5-minute intraday risk monitoring, Sunday full retrain, desktop failure alerts. The book: **7 factors** (volatility 28.8%, high52 22.9%, trend 18.8%, alpha 16.9%, momentum 12.6%, market and volume honestly zero-weighted), 273 point-in-time S&P members, 20-day staggered tranches, inverse-volatility sizing, **broker-side GTC bracket orders** (positions protected overnight and across process crashes), registry-broker reconciliation on every cycle (bracket fires are detected and absorbed automatically — validated live twice in the first fortnight).

Honest expectation for this configuration, from the members-universe evaluation at calibrated costs: **all-period Sharpe ≈ 0.8** (dev 0.75, holdout 1.39), *including* the beta component the attribution in §3.6 decomposes (avg β ≈ 0.59; stock selection ≈ +4.5%/yr of the +13.7% total). Two weeks of live operation surfaced and fixed real bugs (a OneDrive file-lock crash that silently killed one nightly run — now retried with backoff and alerting; an encoding crash in a logging path), exactly what paper trading is for.

## 6. Limitations, stated plainly

1. **Survivorship residual** — delisted members' prices are unavailable with free data; pre-2022 results should be read as upper bounds even after PIT filtering.
2. **Paper fills are a lower bound on costs**; live slippage will be worse than the calibrated 12–15 bp.
3. **The book carries beta — now measured, not just admitted**: average β ≈ 0.59, roughly two-thirds of gross return systematic (§3.6). The Sharpe ≈ 0.8 expectation is not market-neutral performance; the measured stock-selection component is ~+4.5%/yr (IR ~0.5).
4. **Market-feature reproducibility** — nightly rebuilds refetch macro series, so historical feature values can drift slightly between runs (observed: a virgin-tier IC moving 0.0168 → 0.0141); a snapshot cache is the known fix.
5. **Capacity is small** — the strategy trades large-cap US equities in tiny size; measured costs do not extrapolate.
6. **Twenty-one hypotheses is a small family** by industry standards; the ~14% adoption rate and every threshold are computed over exactly the tests recorded, no more, no fewer.

## 7. What I would tell a younger version of this project

- The most valuable line of code in the repo is the purge gap; the most valuable file is the ledger of failures.
- Never let a metric with selection pressure on it (holdout, family threshold) be touched twice.
- When a result looks adoptable, first go find data it has never seen — the past is a cheaper source of virgin data than the future.
- Costs are a hypothesis; fills are data.
- A heuristic you understand beats an optimizer you don't.

*Repository layout, reproduction commands, and per-experiment verdict files are indexed in the README. Every number above traces to a commit.*
