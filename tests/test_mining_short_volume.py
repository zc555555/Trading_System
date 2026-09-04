"""Pins the Reg SHO daily short-volume field (aux_fields.attach_short_volume):
a session's ratio is usable from the next session, a Friday's on Monday,
missing days and uncovered symbols are NaN, sessions after the source's last
covered day are NaN, and future rows never move the past."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=30):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _attach(df, sv, known_through="2024-12-31"):
    return ax.attach_short_volume(df, sv, known_through=known_through)


def _a(out, sym="AAA"):
    sub = out[out.symbol == sym]
    return sub.set_index(sub["date"].dt.tz_localize(None))


def test_next_session_weekend_missing_and_uncovered():
    df = _panel()
    sv = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": ["2024-01-02", "2024-01-05"], "short_ratio_day": [0.4, 0.6]})   # Tue, Fri
    a = _a(_attach(df, sv))
    assert np.isnan(a.loc["2024-01-02", "short_vol_ratio"]) and a.loc["2024-01-03", "short_vol_ratio"] == 0.4
    assert np.isnan(a.loc["2024-01-04", "short_vol_ratio"])            # no row for Wed -> NaN, not carried
    assert a.loc["2024-01-08", "short_vol_ratio"] == 0.6               # Friday's file -> Monday
    assert _a(_attach(df, sv), "BBB")["short_vol_ratio"].isna().all()
    # source known only through 2024-01-05: later sessions are unknown
    a2 = _a(ax.attach_short_volume(df, sv))
    assert a2.loc["2024-01-08", "short_vol_ratio"] == 0.6 and a2.loc["2024-01-09":, "short_vol_ratio"].isna().all()


def test_future_rows_never_move_the_past_and_dsl_sees_it():
    df = _panel()
    base = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-02"], "short_ratio_day": [0.4]})
    fut = pd.concat([base, pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-20"], "short_ratio_day": [0.9]})])
    a, a2 = _a(_attach(df, base)), _a(_attach(df, fut))
    cut = pd.Timestamp("2024-01-19")                                    # the Saturday row lands on Monday 01-22
    np.testing.assert_array_equal(a.loc[:cut, "short_vol_ratio"].to_numpy(), a2.loc[:cut, "short_vol_ratio"].to_numpy())
    out = ax.attach(df, source=NONE, news_source=NONE, form4_source=NONE, short_source=NONE, xbrl_source=NONE,
                    regsho_source=NONE, short_volume=base)
    assert "short_vol_ratio" in out.columns and "short_vol_ratio" in dsl.describe_ops()
    s = dsl.compile_expression("short_vol_ratio * 2", out)
    assert s.notna().sum() == 1                                          # one usable session in `base`


def test_fillna_operator_makes_sparse_daily_fields_windowable():
    df = _panel()
    sv = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": ["2024-01-02", "2024-01-05"], "short_ratio_day": [0.4, 0.6]})
    out = ax.attach_short_volume(df, sv, known_through="2024-12-31")
    raw = dsl.compile_expression("ts_mean(short_vol_ratio, 3)", out)
    filled = dsl.compile_expression("ts_mean(fillna(short_vol_ratio, 0), 3)", out)
    assert raw.notna().sum() == 0 and filled.notna().sum() > 0
    a = filled[out.symbol == "AAA"].to_numpy()
    assert abs(a[4] - (0.4 + 0 + 0) / 3) < 1e-12                      # sessions 01-03..01-05 -> 0.4, 0, 0
    assert "fillna" in dsl.describe_ops()


def test_fillna_keeps_leading_nans_before_coverage_starts():
    df = _panel()
    sv = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": ["2024-01-10", "2024-01-12"], "short_ratio_day": [0.4, 0.6]})
    out = ax.attach_short_volume(df, sv, known_through="2024-12-31")
    f = dsl.compile_expression("fillna(short_vol_ratio, 0)", out)
    a = f[out.symbol == "AAA"].to_numpy()
    assert np.isnan(a[:8]).all()                                         # sessions before 2024-01-11 (index 8): not covered yet
    assert a[8] == 0.4 and a[9] == 0.0 and a[10] == 0.6 and (a[11:] == 0.0).all()   # Wed file -> Thu; Fri file -> Mon
    assert np.isnan(f[out.symbol == "BBB"].to_numpy()).all()             # never covered: stays NaN
