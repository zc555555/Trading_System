"""Pins the freeze semantics of the macro snapshot (research/data/market_snapshot.py).

The reproducibility fix rests on two properties:
  1. a rebuild NEVER changes rows already stored -- even if the upstream
     source now reports different historical values -- it only appends;
  2. the adjusted index is a deterministic function of the stored raw rows
     (dividend/split arithmetic checked against hand-computed values).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from data.market_snapshot import (  # noqa: E402
    adjusted_close, load_snapshot, market_history, update_snapshot,
)


def _fake_source(values_by_date: dict):
    def fetch(symbol, start, end):
        rows = [{"date": d, "close": c, "dividends": 0.0, "splits": 0.0}
                for d, c in sorted(values_by_date.items()) if start <= d <= end]
        return pd.DataFrame(rows, columns=["date", "close", "dividends", "splits"])
    return fetch


def test_rebuild_never_rewrites_stored_rows(tmp_path):
    path = tmp_path / "snap.csv"
    v1 = {"2024-01-02": 100.0, "2024-01-03": 101.0, "2024-01-04": 102.0}
    update_snapshot(["SPY"], "2024-01-04", fetch=_fake_source(v1), path=path,
                    verbose=False)
    # upstream "re-adjusts" history and adds a new day
    v2 = {"2024-01-02": 99.0, "2024-01-03": 100.0, "2024-01-04": 101.0,
          "2024-01-05": 103.0}
    snap = update_snapshot(["SPY"], "2024-01-05", fetch=_fake_source(v2),
                           path=path, verbose=False)
    got = dict(zip(snap["date"], snap["close"]))
    assert got["2024-01-02"] == 100.0 and got["2024-01-04"] == 102.0, \
        "stored history must be frozen"
    assert got["2024-01-05"] == 103.0, "new dates must be appended"
    assert len(snap) == 4
    # persisted, and a failing source leaves the file intact
    def boom(*a):
        raise RuntimeError("yahoo down")
    snap2 = update_snapshot(["SPY"], "2024-01-08", fetch=boom, path=path,
                            verbose=False)
    assert len(snap2) == 4 and len(load_snapshot(path)) == 4


def test_adjusted_close_dividend_and_split_arithmetic():
    rows = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "close": [100.0, 99.0, 25.0, 26.0],       # 4:1 split on 01-04
        "dividends": [0.0, 1.0, 0.0, 0.0],         # $1 ex-div on 01-03
        "splits": [0.0, 0.0, 4.0, 0.0],
    })
    adj = adjusted_close(rows).to_numpy()
    # day 2: price 100 -> 99 plus $1 dividend = flat total return
    assert np.isclose(adj[1], 100.0)
    # day 3: 25 * 4 = 100 vs prev 99 -> +1.0101%
    assert np.isclose(adj[2], 100.0 * (100.0 / 99.0))
    # day 4: 26/25 -> +4%
    assert np.isclose(adj[3], adj[2] * 1.04)


def test_vix_style_symbol_reproduces_raw_close():
    rows = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                         "close": [13.5, 14.2], "dividends": [0.0, 0.0],
                         "splits": [0.0, 0.0], "symbol": "^VIX"})
    view = market_history("^VIX", rows, "2024-01-01", tz=None)
    assert np.allclose(view["close"], view["raw_close"])
    assert np.allclose(view["raw_close"], [13.5, 14.2])


def test_intraday_session_is_never_stored(tmp_path):
    """Running during market hours must not freeze today's partial bar."""
    from data.market_snapshot import last_completed_session
    import pandas as pd
    # 11:00 ET on a Tuesday -> last completed session is Monday
    assert last_completed_session(pd.Timestamp("2026-08-25 11:00", tz="America/New_York")) == "2026-08-24"
    # 16:30 ET -> today counts
    assert last_completed_session(pd.Timestamp("2026-08-25 16:30", tz="America/New_York")) == "2026-08-25"
    path = tmp_path / "snap.csv"
    src = _fake_source({"2026-08-24": 100.0, "2026-08-25": 101.0})
    snap = update_snapshot(["SPY"], "2026-08-26", fetch=src, path=path,
                           verbose=False,
                           now=pd.Timestamp("2026-08-25 11:00", tz="America/New_York"))
    assert list(snap["date"]) == ["2026-08-24"]
