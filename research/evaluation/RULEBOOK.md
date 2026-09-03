# Adoption rulebook v2 — two tracks

*Pre-registered 2026-09-02, before any agent-generated candidate existed.
Code: [`rulebook.py`](rulebook.py). Tests: [`tests/test_rulebook.py`](../../tests/test_rulebook.py).*

## Why the rule changed

The v1 rule was one Šidák family-wise 5% bar on holdout |t| over every
hypothesis in `results/hypothesis_ledger.csv`. It was written for a family
of about twenty human-authored hypotheses with a published mechanism, and it
did its job: it caught a t = 7.5 seasonal mirage and labelled high52's
t = 2.88 as a near-miss instead of a pass.

The next stage of the project lets an LLM agent propose factor expressions
at a rate of hundreds per session, inside the same sealed evaluator (the
AQuA pattern: fixed splits, fixed labels, fixed operator vocabulary, the
agent sees one validation score). Under v1 that family would push the bar
past |t| ≈ 3.5 and forbid adoption by arithmetic, for every candidate, no
matter how good. The rule must be chosen before the first candidate is
generated, or the choice becomes another degree of freedom.

Two prior facts from the ledger motivate two tracks: every data-mined
candidate to date (vol_ext, seasonal, amihud, reversal, investment) died in
a virgin or fresh tier, and the only clean survivor (high52) was
literature-backed. Literature-backed and mined candidates carry different
priors and get different, equally pre-declared, rules.

## Track A — literature (unchanged from v1)

Applies to human-written hypotheses with a published mechanism, tested one
at a time through `experiments/factor_pipeline.py`.

| Item | Rule |
|---|---|
| Family | every row of `hypothesis_ledger.csv` whose verdict does not start with `finding` |
| Gate | holdout \|t\| ≥ Šidák bar(family, α = 5%), two-sided |
| Veto | any tier with t ≤ −2; the fresh tier only once it holds ≥ 60 sessions |
| Removal | pre-registered per factor (high52: negative fresh tier at the 2026-11 review) |

Both `factor_card.py` and `factor_pipeline.py` now compute the family from
this one definition (before v2 the card excluded `finding` rows and the
pipeline did not; the printed bars differed).

## Track B — mined

Applies to agent-generated or otherwise data-mined candidates. Recorded in
`results/mined_candidates.csv`, never in the track-A ledger.

**What the agent sees.** Only the `seen_dev` score of its candidate. The
harness computes `virgin_early`, `holdout` and `fresh` and writes them to
the mined ledger; they are never returned to the agent and never used to
rank candidates during search. (AQuA's split-metric construction, with one
extra hidden tier on each side.)

**One-bit feedback (added 2026-09-03 01:00, user decision, after the three
round-2 full-stage verdicts had been read by the human; it changes what the
agent sees from now on and changes no verdict).** For every candidate that
went through the full stage, the memory document states whether it was
adopted, and nothing else: no tier, no t, no p, no reason. Rationale:
without it the agent cannot learn that a mechanism died on the hidden
tiers and keeps spending budget on it; and non-adoption leaks anyway, since
an adopted factor appears among the incumbents in the agent's prompt. The
leak is one bit per full-stage candidate, and every such candidate is
already a member of the BH family. Code: `mining/memory.py::full_stage_outcomes`.

**Two stages.**

1. *Screen* — standalone per-date rank IC on `seen_dev` only (seconds). Every
   screened candidate is logged with `stage = screen`. Selection here is
   allowed to optimise dev; dev is the only segment tuning may touch.
2. *Full* — the survivor enters the h = 20 purged walk-forward as an extra
   factor group (about 15 minutes) and receives all four tiers. Logged with
   `stage = full`.

**Screen rule (pre-registered 2026-09-02 with the harness).** A screened
candidate is eligible for the full stage when its signed `seen_dev`
Newey–West t (19 lags) is ≥ 2.0 in the declared direction and its dev
coverage is ≥ 0.90. Expressions are limited to the operator registry of
`mining/dsl.py`: windows 1–250, total lookback ≤ 250 sessions, ≤ 40
nodes, depth ≤ 8. Universe: survivorship-complete panel, point-in-time
S&P members, ≥ 40 names per date. Label: 20-session log return.

**Redundancy (screen-rule refinement, added 2026-09-02 21:50 after run
`cc_20260902_round2` screened and before any full-stage verdict of that run
was read).** Every screen pass is compared, by mean per-date Spearman
correlation on `seen_dev`, with a fixed list of incumbent production
features and with every earlier representative pass. Pairs at |ρ| ≥ 0.7
form a cluster; one representative per cluster (an earlier run's
representative keeps precedence, otherwise the highest |dev t| in the
batch); a pass whose best match is an incumbent feature is never a
representative. Only representatives enter the full stage (`full` refuses
the rest without a human `--force`). A residual t (candidate ranks
orthogonalised per date against the most-correlated reference) is reported
to the agent for information; it is not a gate. Rationale: eleven look-alike
passes are one hypothesis, and each full stage adds a member to the BH
family. Code: `mining/redundancy.py`; incumbent list: `INCUMBENT_REFS`.

**Fields (recorded 2026-09-03, after three runs on OHLCV only).** The DSL
gained three point-in-time auxiliary fields from SEC EDGAR filings
(`marketcap`, `turnover`, `filing_days`; see `mining/aux_fields.py`). A
field addition changes the search space, not any rule: screen and gate are
unchanged and the track-B family keeps counting across the change.

**Quarantine (process oracles, added 2026-09-03).** The harness runs
runtime self-checks on every screened candidate (`mining/oracles.py`):
future-perturbation, strength band (|dev t| > 6 or |dev IC| > 0.08),
independent point-in-time mask agreement, label controls, and a
visibility canary. Any firing quarantines the candidate: it cannot enter
the full stage without a human audit and `--force`, and it is never a
cluster representative. These are integrity checks, not adoption rules;
their detection power is measured by the mutation-injection experiment.

**Horizons (pre-registered 2026-09-03 16:00, user decision, before any
5-session candidate existed).** Track B runs one family per label
horizon: h = 20 (production) and h = 5. Same screen, same gate, same
oracles; the label is the h-session log return and Newey–West uses
h − 1 lags. Duplicates, representatives and the BH family are all
per horizon: the same expression at another horizon is a different
test. An h = 5 adoption is a *shelf* entry (`status: shelf` in the
registry): production consumers compile only production-horizon
entries, so nothing validated at h = 5 can reach the live book until an
h = 5 book exists (gated on the execution A/B, ~2026-11) and its own
adoption is decided then. Rationale: the only honest signals ever
measured at t > 2 on this ruler were at h = 5 (trend, volatility, news).

**Gate (all clauses required).**

| Clause | Rule |
|---|---|
| Direction | `expected_direction` is declared in the proposal before evaluation; holdout IC must have that sign. No post-hoc sign calibration. |
| Family | every track-B candidate that reached the full stage, cumulative over the project |
| Test | one-sided holdout p in the declared direction, Benjamini–Hochberg FDR **q = 0.10** over the family |
| Dev gain | blended `seen_dev` IC with the candidate > without it |
| Sample | holdout ≥ 120 sessions |
| Veto | any tier with signed t ≤ −2 in the declared direction; fresh only once it holds ≥ 60 sessions |

A verdict is computed over the family as it stands at adjudication and
recorded with `family_n` and the BH threshold. It is not re-adjudicated as
the family grows, except at a scheduled review (next: 2026-11).

**Why these numbers.** q = 0.10 accepts that one in ten track-B adoptions
may be a false discovery; the fresh-tier veto and the 2026-11 review are
the second line. The direction clause converts AQuA's direction-calibration
step (which we consider a selection channel) into a falsification test and
buys a one-sided p honestly. The dev-gain clause is the same rule that
adopted high52. 120 holdout sessions is the shortest window on which a
20-day-overlap Newey–West t has been trusted in this project.

## Forbidden under either track

- Choosing q, α, or the family definition after seeing results.
- Re-screening or ranking candidates on holdout, virgin or fresh.
- Moving a candidate between tracks after its scores are known. A mined
  expression that later acquires a literature citation stays in track B.
- Reporting a track-B result under the track-A bar, or vice versa.
- Any LLM call in the live signal path. Adoption is a human decision made
  on the hidden tiers; the agent proposes, the harness scores, the ledger
  records.

## Expected outcome, stated in advance

At h = 20 on this universe honest single-factor ICs are 0.01–0.03 with
t ≈ 2 on ~100 effectively independent 20-day windows. The prior from the
ledger is a 14% adoption rate for hand-picked candidates; mined candidates
should do worse. Zero to two track-B adoptions per hundred full-stage
candidates would be consistent with the record. A materially higher rate
is itself a warning to audit the harness for leakage before believing it.
