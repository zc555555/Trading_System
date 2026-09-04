# Agent factor mining — harness protocol (rulebook track B)

The mining agent is a proposer. It never touches data, the evaluator, the
label, the splits, or production. Its whole interface is one JSON file in
and one JSON blob out. Rules: [`evaluation/RULEBOOK.md`](../evaluation/RULEBOOK.md).

## What the agent may do

1. Read the operator registry: `python mining/harness.py ops`.
2. Write a proposals file (see [`examples/proposals_smoke.json`](examples/proposals_smoke.json)).
   Every proposal carries, **before any score is seen**:
   - `candidate_id` — `^[a-z][a-z0-9_]{2,30}$`, unique for the project
   - `expression` — one line in the DSL of [`dsl.py`](dsl.py)
   - `expected_direction` — `positive` (higher value → higher 20-session return) or `negative`
   - `hypothesis`, `mechanism` — sentences, not labels
   - `refutation_conditions` — at least one
3. Screen: `python mining/harness.py screen proposals.json` → per candidate
   `dev_ic`, `dev_t`, `dev_n`, `coverage`, `lookback`, `n_nodes`, `screen_pass`.
   Cheap (seconds per candidate after the first run builds a cache).
4. Full stage, only for `screen_pass = true`: `python mining/harness.py full
   proposals.json --id CID` (about 15–25 minutes). Returns `dev_ic`, `dev_t`,
   `blend_dev_gain` (blended seen_dev IC with the candidate minus without).
   Nothing else.

## Auxiliary fields (SEC EDGAR, GDELT news, Form 4, FINRA short interest; point-in-time)

Insider trades: `insider_buys` / `insider_sells` (distinct insiders whose
open-market purchase / sale Form 4s became usable at this session, 0 when the
symbol is covered and quiet, NaN when it has no SEC mapping or after the
dataset's last covered day) and `insider_net_frac` (shares bought minus sold,
over the EDGAR share count). Source: SEC Insider Transactions data sets
(quarterly zips in `data/form345/`, `data/build_form4_fields.py` ->
`data/form4_daily.parquet`; only original Form 4s, TRANS_CODE P and S). A
filing on day D is usable from the first session strictly after D.

Short interest: `short_ratio` (latest public FINRA consolidated short
interest / EDGAR share count) and `days_to_cover` (FINRA's own). Source:
`data/fetch_finra_short_interest.py` -> `data/finra_short_interest.parquet`
(FINRA Query API, semi-monthly settlement dates from 2017-12-29). A figure
settled on session s is usable from s + `SHORT_INTEREST_LAG` (10) sessions
(FINRA publishes about seven business days after settlement) and expires
after `SHORT_MAX_AGE` (40) sessions.


News: `news_tone` (article-weighted mean GDELT tone of the calendar days that
became usable at this session, NaN when no article) and `news_articles`
(article count, 0 when the symbol is covered but quiet, NaN when the symbol is
not in the source). Source: `data/gdelt_bigquery.py --names
data/company_names_full.json` -> `data/gdelt_daily.parquet` (BigQuery
`gdeltv2.gkg_partitioned`, ~260 GB per full 2017-2026 pass). Name matching
follows GDELT's own spelling (apostrophes and possessives dropped, corporate
suffixes stripped on both sides, `tests/test_gdelt_names.py`); an article
counts once per symbol however many times it names the company. GDELT's
organisation tagger never emits some large names (State Street, Capital One,
AT&T, 3M, Phillips 66, Fifth Third), which therefore stay NaN. A calendar
day's news is usable from the first session strictly after it, so
Friday-Sunday news lands on Monday.

Beyond OHLCV the DSL exposes `marketcap` (close x latest usable cover-page
share count, USD millions), `turnover` (volume / share count) and
`filing_days` (sessions since the last 10-Q/10-K filing usable today). Source:
`data/build_edgar_fields.py` -> `data/edgar_fields.parquet`, built from the
local EDGAR companyfacts cache (`dei:EntityCommonStockSharesOutstanding`, one
row per filing with its `filed` date). A filing on session a is usable from
a + 1 (the PEAD experiment's convention); a share count older than 130
sessions expires. Attached by [`aux_fields.py`](aux_fields.py) to the screen
cache, the full-stage panel and, for adopted factors, the production feature
build. An expression that uses a field whose source file is missing is
refused, never zero-filled. Adopting a factor that uses them requires
`fetch_fundamentals.py` + `build_edgar_fields.py` to run before the nightly
feature build (not wired yet; see PRODUCTION.md). Sharadar's DAILY/EVENTS
tables were tried first and are NOT in the Prices subscription (HTTP 403
beyond the free DJIA-30 sample). Tests: `tests/test_mining_aux_fields.py`.

## Running the agent (step 3: technical isolation)

```bash
cd research
python mining/agent.py --dry-run                      # scripted fake model, no API call
python mining/agent.py --goal "..." --max-screens 40  # real run (claude-opus-5 by default)
python mining/agent.py --run-full RUN_ID              # afterwards: queued full stages
```

`agent.py` runs the model with exactly four client-side tools (list_operators,
screen_proposals, request_full_stage, submit_notes) and nothing else: no file,
shell or network tool. Every tool result is built by the harness and filtered
through `agent_view`; the full stage never runs inside the session, so hidden
tiers do not exist in the process while the model is talking. The transcript
of every request, tool call and result is `mining/runs/<run_id>/transcript.jsonl`
(git-ignored) for audit. Credentials: `ANTHROPIC_API_KEY` in the environment or
in `config_keys.py`. Server-side refusal fallbacks are on by default
(`--no-fallbacks` to disable).

## Running the agent inside Claude Code (no API billing)

The same protocol runs as a Claude Code subagent, billed to the Claude
subscription instead of the API. Definition: [`.claude/agents/miner.md`](../../.claude/agents/miner.md)
(tools: Bash, Write only). Enforcement: [`.claude/hooks/miner_guard.py`](../../.claude/hooks/miner_guard.py),
registered as a PreToolUse hook on every tool in `.claude/settings.json`.
It keys on the `agent_type` field Claude Code puts in the hook input (the
model cannot set it) and is default-deny for the miner:

| Tool | Allowed for the miner |
|---|---|
| Bash | `research/venv/Scripts/python.exe research/mining/harness.py ops` |
| Bash | `... harness.py screen research/mining/runs/cc_<run>/proposals[_N].json [--out <same run>/screen[_N].json]` |
| Write | `research/mining/runs/cc_<run>/{proposals[_N].json, full_queue.json, notes.md}`; a proposals file is refused once the run would exceed 40 candidates |
| anything else | denied (Read, Grep, Glob, Edit, other Bash, other paths) |

Every decision for the miner is appended to `research/mining/runs/guard.log`.
Launch from a Claude Code session in this repo (the agent type loads at
session start):

```
Agent(subagent_type="miner", prompt="Run id: cc_20260903_a. Research goal: ...")
```

then, as the human: `python mining/agent.py --run-full cc_20260903_a` for the
queued full stages, and `python mining/harness.py show <id>` to adjudicate.
Tests: `tests/test_miner_guard.py`.

## Cross-run memory (step 4)

The miner cannot read files, so its memory is a document the harness builds
([`memory.py`](memory.py)) and prints via `harness.py memory` (allowed by the
guard). Three sources, dev-stage only:

1. `beliefs.json` written by the agent at the end of each run (one entry per
   mechanism: status `dead|weak|promising|untested`, evidence, next), merged
   across runs with the latest run winning per mechanism;
2. the index of every canonical expression ever screened, with its declared
   direction and dev statistics. A re-submitted expression is not re-scored:
   `screen` returns the old record with `duplicate_of`, re-applies the pass
   rule in the new proposal's declared direction, and writes no ledger row;
3. the last two runs' `notes.md`, verbatim, truncated.

Hidden tiers, verdicts and the BH family never enter the document (test:
`tests/test_mining_memory.py` plants hidden numbers in a ledger and asserts
they are absent).

**Redundancy clustering (screen-rule refinement).** Every screen pass is
compared, by mean per-date Spearman correlation on the development segment,
with the incumbent feature list in [`redundancy.py`](redundancy.py), every
earlier representative pass and the rest of its batch. |ρ| ≥ 0.7 joins a
cluster; one representative per cluster (earlier run's representative keeps
precedence, else highest |dev t|); a pass that restates an incumbent feature
is never a representative. Only representatives may enter the full stage
(`full` refuses the rest without `--force`). The agent sees `cluster_rep`,
`redundant_with`, `max_corr`, `corr_with` and `residual_dev_t` (candidate
orthogonalised per date against its closest reference, re-tested). Test:
`tests/test_mining_redundancy.py`.

**Reflection and learning audit.** Before its first proposal a run must
write `reflection.md` (what failed last time and why, what it will not touch,
which weak/untested leads it follows and how, what it changes procedurally).
Every proposal carries a `mechanism_tag` matching a beliefs entry. Afterwards
`python mining/harness.py audit-run <run_id>` (human) reports whether the run
aimed proposals at mechanisms earlier runs had marked dead, re-submitted
indexed expressions, wrote reflection and beliefs, and which beliefs changed
status; `learned` is true only when all four hold. Test: `tests/test_mining_audit.py`.

## Process-level oracles (runtime self-checks)

[`oracles.py`](oracles.py) runs on every screened candidate, whatever
produced the feature: a metamorphic **future-perturbation** check (rows
after a cutoff are perturbed; pre-cutoff values must not move), a
**strength** plausibility band (|dev t| > 6 or |IC| > 0.08 is an audit
signal, not a discovery), an **N-version membership** check (the dev IC is
recomputed on an independently derived point-in-time mask and must match),
batch-level **controls** (expressions with a known honest IC band are
screened alongside; a misaligned label throws them outside it), and a
**visibility canary** (no hidden key or number may appear in the text handed
to the agent; the harness refuses to emit rather than leak). A firing
quarantines the candidate (`quarantined`, `oracle_flags`, never eligible for
the full stage without a human `--force`) and is logged to
`mining/runs/oracle.log`.

The mutation-injection experiment
(`evaluation/experiments/mutation_injection.py`) injects six defect classes
with known real instances around the harness and records which oracle
catches each one, plus the false-alarm count on a clean run; report in
`evaluation/results/mutation_injection_report.md`. Tests:
`tests/test_mining_oracles.py`.

## What the agent never sees

`virgin_early`, `holdout`, `fresh` tiers, the one-sided holdout p, the
Benjamini–Hochberg threshold, the family size, the verdict and its reasons.
They are written to `evaluation/results/mined_candidates.csv` and shown only by
`python mining/harness.py show CID`, a human command. The agent must not read
that file or any `evaluation/results/oos_predictions_h20_mined_*.parquet`
(step 3 of the plan makes this a tool restriction rather than an instruction).

## What "usable in production" means

A human runs `python mining/harness.py adopt CID --removal-trigger "..."`,
which is refused unless the latest full-stage verdict is PASS. Adoption writes
one entry to `factors/mined_factors.json`; from then on the expression is
compiled by `features/build_dataset.py` on every nightly rebuild, z-scored
with the other features, registered as its own factor group, and trained by
the Sunday retrain like the seven incumbent factors. Then:

```bash
cd research
python prepare_prediction_data.py                                   # features incl. mined_<id>
python evaluation/experiments/survivorship_universe.py --reuse-panel  # new track-B baseline
python run_retrain_models.py                                        # 8+ factor ensembles
```

## Calibration note (2026-09-02 smoke)

On the real panel the raw DSL restatement of the project's best incumbent
factor, `close / ts_max(high, 250)`, screens at dev IC +0.010 (t 0.39), the
same as the production column `pct_52w_high` scored independently (+0.0097,
t 0.37). The factor card's +0.032 for high52 is the *trained* group score
(two features, cross-sectional z-scores, three tree models). A raw expression
that passes the screen (signed t ≥ 2.0) therefore carries more standalone
signal than any incumbent feature does; the bar is high by design.
