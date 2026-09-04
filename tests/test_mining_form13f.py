"""Pins the 13F institutional-ownership fields (aux_fields.attach_form13f):
a quarter is usable from the first session strictly after its 45-day
deadline, expires after INST_MAX_AGE sessions, uses the EDGAR share count,
is NaN before a symbol's first quarter, and the next quarter never moves
the past."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=200):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


FILINGS = pd.DataFrame({"symbol": ["AAA", "BBB"], "filed": ["2023-12-20", "2023-12-20"], "shares_out": [1_000_000.0, 1_000_000.0]})


def _attach(df, q):
    return ax.attach(df, source=NONE, filings=FILINGS, news_source=NONE, form4_source=NONE, short_source=NONE,
                     xbrl_source=NONE, regsho_source=NONE, form13f_source=NONE, form13f=q)


def _a(out, sym="AAA"):
    sub = out[out.symbol == sym]
    return sub.set_index(sub["date"].dt.tz_localize(None))


def test_deadline_timing_ratio_and_expiry():
    df = _panel()
    q = pd.DataFrame([{"symbol": "AAA", "period_end": "2023-12-31", "usable_from": "2024-02-14",     # Q4-2023, deadline Wed 02-14
                       "inst_shares": 600_000.0, "inst_holders": 120.0, "top5_share": 0.3, "n_filings": 5000.0}])
    a = _a(_attach(df, q))
    assert np.isnan(a.loc["2024-02-14", "inst_own"])                    # deadline day itself: not yet
    assert abs(a.loc["2024-02-15", "inst_own"] - 0.6) < 1e-12
    assert a.loc["2024-02-15", "inst_holders"] == 120 and abs(a.loc["2024-02-15", "inst_top5"] - 0.3) < 1e-12
    sessions = list(a.index)
    i = sessions.index(pd.Timestamp("2024-02-15"))
    assert not np.isnan(a.iloc[i + ax.INST_MAX_AGE]["inst_holders"])
    assert np.isnan(a.iloc[i + ax.INST_MAX_AGE + 1]["inst_holders"])   # one quarter plus slack, then unknown
    assert _a(_attach(df, q), "BBB")[["inst_own", "inst_holders", "inst_top5"]].isna().all().all()


def test_next_quarter_never_moves_the_past_and_dsl_sees_fields():
    df = _panel()
    q1 = pd.DataFrame([{"symbol": "AAA", "period_end": "2023-12-31", "usable_from": "2024-02-14",
                        "inst_shares": 600_000.0, "inst_holders": 120.0, "top5_share": 0.3, "n_filings": 5000.0}])
    q2 = pd.concat([q1, pd.DataFrame([{"symbol": "AAA", "period_end": "2024-03-31", "usable_from": "2024-05-15",
                                        "inst_shares": 700_000.0, "inst_holders": 130.0, "top5_share": 0.25, "n_filings": 5000.0}])])
    a, a2 = _a(_attach(df, q1)), _a(_attach(df, q2))
    cut = pd.Timestamp("2024-05-15")
    for c in ("inst_own", "inst_holders", "inst_top5"):
        np.testing.assert_array_equal(a.loc[:cut, c].to_numpy(), a2.loc[:cut, c].to_numpy())
    assert a2.loc["2024-05-16", "inst_holders"] == 130
    out = _attach(df, q2)
    s = dsl.compile_expression("delta(inst_holders, 63)", out)
    assert s.notna().sum() > 0
    for f in ("inst_own", "inst_holders", "inst_top5"):
        assert f in dsl.AUX_FIELDS and f in dsl.describe_ops()
