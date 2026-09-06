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

**Fields (recorded 2026-09-04, after five runs; h=20 n=3 and h=5 n=2 full
stages, no pass).** Two GDELT news fields were added (`news_tone`,
`news_articles`; source `data/gdelt_bigquery.py` -> `data/gdelt_daily.parquet`,
803 members, 2017-01-01 onward; one count per article and symbol). Calendar day D's news is usable from the
first session strictly after D (weekend news lands on Monday), one session
more conservative than the news experiment's same-day convention. Same
principle as above: a search-space change, no rule change, families keep
counting. Coverage before 2017 is NaN, so virgin_early evidence for news
factors is thinner than for OHLCV factors; the pooled/recent tests are
unaffected because they use only the sessions where the factor exists.

**Fields (recorded 2026-09-04, after round 6: the news representative failed
the hidden tiers).** Five more fields: `insider_buys`, `insider_sells`,
`insider_net_frac` (SEC Form 4 open-market trades, usable from the session
after filing, NaN after the quarterly dataset's last covered day) and
`short_ratio`, `days_to_cover` (FINRA consolidated short interest, usable ten
sessions after settlement, from 2018). Again a search-space change only.
Short-interest evidence before 2018 does not exist, so its virgin_early tier
is the shortest of any field family (2018 to 2021-12-30).

**Review procedure (recorded 2026-09-04; first review 2026-11-15).** Mining
rounds are paused after round 8 (eight runs, zero adoptions; the
short-interest level family is the only mechanism significant on seen_dev
and holdout at both horizons, failing once on blend gain and once on a
37-session fresh tier). On each REVIEW_SCHEDULE date
`scripts/run_review_scheduled.bat <date>` refreshes the surv panel, re-runs
the incumbent baseline and every track-B full-stage candidate at both
horizons, and `harness.py --horizon H review --date <date> --rerun`
re-adjudicates them under `segments_for(date)` as ONE BH family per horizon
(each candidate's pooled and recent p, everyone else's as its family). The
rotation is applied at runtime; nothing is written to the ledger and nothing
is adopted by the script. A PASS in `results/review_<date>_h<H>.md` is acted
on by hand: set `ACTIVE_REVIEW` to the date, commit, re-run the candidate
with `full --force` (its ledger row is then under the rotated segments) and
`adopt`. A dry review (no `--rerun`) re-scores the recorded out-of-sample
predictions and is exact for the raw-feature tiers. External sources
refresh monthly through `scripts/run_refresh_sources_scheduled.bat`.

## Universes (pre-registered 2026-09-06)

Two research universes, each with its own track-B family per horizon, its
own screen caches, pool and memory; the ledger records `universe`.

* `sp500`: the survivorship-complete point-in-time S&P 500 panel. The only
  universe whose adoptions can reach the live book.
* `midcap`: US domestic common stocks ranked 501-1400 by market cap at a
  month end enter, and leave below 450 or above 1650 or on delisting
  (`data/build_midcap_universe.py`). Market cap = Sharadar month-end close
  x the latest SEC frames share count whose quarter end is at least 45 days
  old (the frames API carries no filing date; the statutory deadline stands
  in). Delisted names are ranked while they trade, so the universe contains
  the dead. Research only: nothing mined here is adopted into production;
  a factor that passes on mid caps is evidence for a later S&P test, not a
  substitute for it. Its auxiliary fields are the subset with a mid-cap
  source (share count from frames, daily short volume, sectors); absent
  sources are absent fields, never zero.

Rationale: on S&P 500 large caps every honest signal has |IC| 0.01-0.02
and the pool saturated at ~25 members after ~280 candidates; the anomaly
literature's base rates are higher outside the largest 500 names.

## Track P: the factor pool (v3.1, pre-registered 2026-09-04)

Motivation. Nine rounds of single-candidate adjudication produced zero
adoptions: on S&P 500 large caps every honest signal has |IC| 0.01-0.02
and no single one clears a hidden-tier gate. The industry answer is
breadth: many weak, low-correlation signals combined and tested as one
object. Track P adds that path without weakening track B.

Admission (development segment only; nothing hidden is consulted):
  * screen pass in the declared direction (dev t >= 2.0, coverage >= 0.90),
    no process-oracle flag, not a duplicate;
  * not redundant with a production feature (|rho| < 0.7, the existing
    screen rule); |rho| < 0.7 with every current pool member;
  * residual dev t against the current composite >= 1.5 in the declared
    direction (the first member has no composite to beat).
  Screen output carries pool_size, pool_corr_max, pool_corr_with and
  residual_vs_pool_t so the miner can target information the pool lacks.

Composite. Per member: cross-sectional percentile rank per date, centred,
times the declared sign. Composite = plain mean over members with a value
(at least 30% of members non-missing). Equal weights by design.

Release and adoption. At member counts 25, 50, 100, 200, 400 the composite
is scored exactly like a track-B full-stage candidate (purged walk-forward
as its own factor group, raw-feature tiers, v3 two-path gate, blend gain)
under the ledger id pool_h<H>_r<k>. EACH RELEASE IS ONE MEMBER OF THE
HORIZON'S BH FAMILY; individual pool members never enter the family. A
PASS adopts the composite as one production factor; members are then
frozen for that release and later admissions form the next release.

Honesty notes. The pool's development-segment IC is not evidence (the
development segment is reused by every admission); only release rows are.
A release that fails does not remove members; the pool keeps growing and
the next checkpoint is the next test. The segment-rotation review
re-adjudicates release rows like any full-stage row.

**v3.2 admission (pre-registered 2026-09-04, user decision after two pool
rounds admitted 4 of 80 candidates).** The pool's own bar replaces the
track-B screen pass: signed dev t >= 1.0 in the declared direction,
coverage >= 0.90, no oracle flag, |rho| < 0.6 with every incumbent feature
and every member, residual dev t against the composite >= 1.0. Rationale:
the tested object is the composite, whose noise shrinks with the square
root of the number of uncorrelated members; a member is an input, not a
claim. The release test is unchanged, and the development-segment IC of
the pool is explicitly not evidence. Revert to the v3.1 bar if the first
releases fail (the user's stated condition). Members record
`rule_version` P-3.2.

**Operators (2026-09-04).** `sector_rank(x)` and `sector_demean(x)`:
cross-sectional operators within (date, sector); the sector label comes
from Sharadar's classification (Yahoo's map as fallback, 'Unknown' never
guessed). Search-space change only.

**Fields (recorded 2026-09-04, mining paused).** Thirteen more fields while
rounds are paused: twelve point-in-time accounting ratios from the EDGAR
XBRL cache (`book_to_market` ... `op_margin`, see mining/README) and the
FINRA Reg SHO daily short-volume ratio `short_vol_ratio` (from 2018), plus
one elementwise operator `fillna(x, c)` (where() keeps NaN conditions NaN,
so sparse daily fields could not be windowed before).
Later the same day: `inst_own`, `inst_holders`, `inst_top5` from the SEC
Form 13F data sets (usable from the session after the 45-day deadline).
And `wiki_views` (English-Wikipedia page views, usable from the session
after the UTC day). Twenty-seven auxiliary fields in total.
Search-space change only; no full-stage candidate is added before the
2026-11-15 review, so the BH families are unchanged. New sources are checked
with screen-stage runs that are NOT recorded in the ledger (`screen
--no-record`) so that a textbook expression tried by hand does not enter a
family. Any mined candidate on these fields after the review joins the
family of its horizon as usual.

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

**v3 (pre-registered 2026-09-04, user decision, after five track-B runs and
five full-stage verdicts had been read; applies to candidates adjudicated
from now on; earlier verdicts stand until the 2026-11-15 review, when every
track-B full-stage candidate is re-adjudicated under v3 on the rotated
segments and the family is recomputed).** The two-track structure, the
screen, the oracles and the one-bit feedback are unchanged. What changes:

*Two adoption paths, one family.* Every full-stage candidate contributes
two one-sided p-values (declared direction, Newey–West h − 1 lags) to the
BH family at q = 0.10: the p of its per-date IC pooled over every unseen
tier (virgin_early + holdout + fresh), and the p of its IC pooled over
the recent tiers (holdout + fresh). FDR is therefore controlled over all
tests actually run. Common clauses for either path: holdout ≥ 120
sessions, blended seen_dev gain > 0, no unseen tier significantly
negative in the declared direction (fresh vetoes once it holds ≥ 60
sessions).

| Path | Test | Also required | Adoption |
|---|---|---|---|
| structural | pooled p rejected by BH, and pooled t ≥ 2 in the declared direction (BH alone can let a candidate's strong recent p carry a weak pooled p; the label must be earned by the pooled evidence itself) | ≥ 2 of the available unseen tiers positive (all if fewer than 2); recent IC not negative | normal, with the usual pre-registered removal trigger |
| probation | recent p rejected by BH | holdout positive, fresh positive when available | adopted under probation: the IC monitor's first WARN removes it, static weight capped at half |

Rationale. A single ~250-session holdout is one regime and the verdicts
hinged on whether an effect happened to be present in it; pooling
multiplies the unseen sample about five-fold. But a rule that only
admits effects alive since 2015 would never admit a young one (high52,
the one factor that survived track A, is flat in virgin_early), and
alpha decays, so demanding significance across a decade is the wrong
shape. The probation path admits a young effect on its post-dev
evidence and prices the shorter history through monitoring rather than
through a higher bar. seen_dev stays the only selection segment; nothing
unseen is shown to the agent.

*Rotation.* At each review date in `rulebook.REVIEW_SCHEDULE` (2026-11-15,
then every six months) the segments move: holdout becomes the twelve
months before the review date, fresh becomes everything after it, the old
holdout joins seen_dev; virgin_early never changes. A rotation is executed
by setting `rulebook.ACTIVE_REVIEW` to that date and nothing else. Every
verdict reveals holdout information to the human who sets the next goal,
so a holdout window wears out; the newest data must always be the sealed
one. The agent therefore mines data that is six to twelve months old —
that lag is the price of verifiability.

Operations: the survivorship-complete panel is rebuilt monthly so the
fresh tier keeps growing (PRODUCTION.md).

**Gate (v2 wording, kept for the record of how the first five full stages were judged; v3 above is current).**

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
