# Contributing

This is primarily a personal trading-research project, but if you're reading
this and want to send a patch, the conventions below keep things consistent.

## Development setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
```

## Code style

- **Formatter:** [`black`](https://black.readthedocs.io/) (line length 100)
- **Linter:**    [`ruff`](https://docs.astral.sh/ruff/) (config in `pyproject.toml`)
- **Types:**     [`mypy`](https://mypy-lang.org/) — types are encouraged but
  not yet enforced project-wide.

Run all three before opening a PR:

```bash
ruff check .
black --check .
mypy .
```

To auto-fix what can be auto-fixed:

```bash
ruff check --fix .
black .
```

## Tests

```bash
pytest
```

- Unit tests live in `tests/` (TODO: scaffold).
- Anything that hits Alpaca / Finnhub must be marked `@pytest.mark.integration`
  and is skipped by default in CI.
- Anything taking more than a few seconds is `@pytest.mark.slow`.

The existing `test_alpaca_connection.py` and `test_finnhub.py` at the project
root are **smoke tests**, not pytest unit tests — they hit the real APIs and
print human-readable output. Run them manually when validating credentials or
network reachability.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

Common types:

| Type     | Use for                                      |
|----------|----------------------------------------------|
| `feat`   | New feature                                  |
| `fix`    | Bug fix                                      |
| `refactor`| Code change that neither fixes nor adds      |
| `docs`   | Documentation only                           |
| `chore`  | Tooling, build, dependencies                 |
| `test`   | Adding / fixing tests                        |
| `perf`   | Performance improvement                      |

Example:

```
feat(monitor): add account-wide intraday loss kill switch

When cumulative day P&L drops below MAX_DAILY_LOSS_PCT, flatten all
positions and skip further entries until the next session.
```

## Branching

- `main` — stable, paper-tradable.
- `feature/<short-name>` — feature work.
- `fix/<short-name>` — bug fixes.

Open a PR back into `main`. Self-merge is fine; just keep the message clean.

## What NOT to commit

- API keys (`config_alpaca.py`, `config_finnhub.py`) — already in `.gitignore`.
- Trained model artefacts (`research/artifacts/*.pkl`) — too large; regenerate
  with `python run_retrain_models.py`.
- Runtime output (`trading_logs/`, `paper_trading_reports/`, `alerts/`,
  `news_cache/`) — already gitignored.
- Virtual environments (`venv/`, `research/venv/`) — already gitignored.
