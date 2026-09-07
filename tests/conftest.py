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
