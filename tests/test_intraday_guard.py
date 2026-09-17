"""trading/intraday_guard.py: the only intraday risk action left after the
brackets were removed, on FakeBroker."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading import halt, intraday_guard as ig  # noqa: E402
from trading.broker import FakeBroker  # noqa: E402
from trading.trading_calendar import EASTERN  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731
MIDDAY = datetime(2026, 9, 16, 12, 0, tzinfo=EASTERN)


def _paths(tmp_path):
    return dict(halt_path=tmp_path / "halt.json", failures_log=tmp_path / "F.log", guard_log=tmp_path / "g.log")


def test_session_gate():
    assert ig.in_session(MIDDAY)
    assert not ig.in_session(datetime(2026, 9, 16, 9, 29, tzinfo=EASTERN))
    assert not ig.in_session(datetime(2026, 9, 16, 16, 0, tzinfo=EASTERN))
    assert not ig.in_session(datetime(2026, 9, 7, 12, 0, tzinfo=EASTERN))        # Labor Day
    assert not ig.in_session(datetime(2026, 9, 13, 12, 0, tzinfo=EASTERN))       # Sunday


def test_no_action_within_the_band_and_outside_the_session(tmp_path):
    b = FakeBroker(equity=98_000, last_equity=100_000, prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    out = ig.run_once(b, MIDDAY, 0.03, sleep=NOSLEEP, **_paths(tmp_path))
    assert out["action"] == "ok" and abs(out["loss_pct"] + 2.0) < 1e-9 and b.pos == {"AAA": 10.0}
    b2 = FakeBroker(equity=90_000, last_equity=100_000, prices={"AAA": 100})
    b2.pos = {"AAA": 10.0}
    out2 = ig.run_once(b2, datetime(2026, 9, 16, 17, 0, tzinfo=EASTERN), 0.03, sleep=NOSLEEP, **_paths(tmp_path))
    assert out2["action"] == "skip_closed" and b2.pos == {"AAA": 10.0} and not halt.is_halted(tmp_path / "halt.json")


def test_breach_halts_cancels_and_flattens_once(tmp_path):
    b = FakeBroker(equity=96_500, last_equity=100_000, prices={"AAA": 100, "SSS": 50})
    b.pos = {"AAA": 10.0, "SSS": -4.0}
    b.submit("BBB", 5, "buy", kind="limit", limit_price=10.0, client_order_id="open_t_BBB")
    p = _paths(tmp_path)
    out = ig.run_once(b, MIDDAY, 0.03, sleep=NOSLEEP, timeout_s=2, poll_s=0.01, **p)
    assert out["action"] == "halted" and halt.is_halted(p["halt_path"])
    st = halt.read(p["halt_path"])
    assert st["source"] == "intraday_guard" and "daily loss -3.50%" in st["reason"]
    assert b.pos == {} and out["flatten"]["canceled"] >= 1 and out["flatten"]["unfilled"] == []
    assert "HALT set by intraday guard" in (tmp_path / "F.log").read_text(encoding="utf-8")
    # the next tick does nothing (already halted): no second flatten, no new orders
    n_calls = len(b.calls)
    out2 = ig.run_once(b, MIDDAY, 0.03, sleep=NOSLEEP, **p)
    assert out2["action"] == "skip_halted" and len(b.calls) == n_calls


def test_broker_outage_fails_safe(tmp_path):
    b = FakeBroker(equity=90_000, last_equity=100_000, prices={"AAA": 100})
    b.pos = {"AAA": 10.0}
    b.fail_next["account"] = 1
    p = _paths(tmp_path)
    out = ig.run_once(b, MIDDAY, 0.03, sleep=NOSLEEP, **p)
    assert out["action"] == "broker_error" and b.pos == {"AAA": 10.0} and not halt.is_halted(p["halt_path"])
    assert "broker unavailable" in (tmp_path / "g.log").read_text(encoding="utf-8")
