import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make both repo-root modules (trading/) and research-layer modules importable
for p in (ROOT, ROOT / "research"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _no_desktop_notifications(monkeypatch):
    """A halt set inside a test must not raise a real desktop toast; what
    would have been sent is inspectable through trading.halt._TEST_SENT."""
    try:
        from trading import halt as _halt
    except Exception:                                # noqa: BLE001
        return
    sent = []
    monkeypatch.setattr(_halt, "NOTIFY", lambda title, body: sent.append((title, body)))
    monkeypatch.setattr(_halt, "_TEST_SENT", sent, raising=False)


@pytest.fixture(autouse=True)
def _membership_table_when_absent(tmp_path, monkeypatch):
    """CI checks out tracked files only, so research/data/sp500_membership.parquet
    (weekly Wikipedia refresh, gitignored) is absent there and the membership
    oracle raised FileNotFoundError inside screen_one, turning every screened
    candidate into an error (first seen on the 2026-09-07 push). On the
    synthetic panels the tests screen, no symbol is a real member anyway, so
    the oracle is vacuous locally: give it an EMPTY table when the real one is
    missing, and always a private PIT cache so tests never overwrite the
    production cache under research/."""
    try:
        from mining import harness, oracles
    except Exception:                                # noqa: BLE001
        return
    import pandas as pd
    monkeypatch.setattr(oracles, "PIT_INDEPENDENT", tmp_path / "_pit_independent.parquet")

    def _empty_like(path: Path) -> Path:            # same basename: universe tests check MEMBERSHIP.name
        out = tmp_path / Path(path).name
        if not out.exists():
            pd.DataFrame({"symbol": pd.Series(dtype=str), "start": pd.Series(dtype="datetime64[ns]"),
                          "end": pd.Series(dtype="datetime64[ns]")}).to_parquet(out, index=False)
        return out

    for cfg in harness.UNIVERSE_CFG.values():
        if not Path(cfg["membership"]).exists():
            monkeypatch.setitem(cfg, "membership", _empty_like(cfg["membership"]))
    if not Path(harness.MEMBERSHIP).exists():
        monkeypatch.setattr(harness, "MEMBERSHIP", _empty_like(harness.MEMBERSHIP))
