"""Pins the Wikipedia attention field (aux_fields.attach_wikipedia): a UTC
day's views are usable from the next session, Friday-Sunday sum into
Monday, NaN before the article's first day / for symbols without an
article / after the source's last covered day, and future days never move
the past."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=30):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")   # Mon 2024-01-01 ...
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _a(out, sym="AAA"):
    sub = out[out.symbol == sym]
    return sub.set_index(sub["date"].dt.tz_localize(None))


def test_next_session_weekend_sum_and_coverage_rules():
    df = _panel()
    pv = pd.DataFrame({"symbol": ["AAA"] * 5, "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06", "2024-01-07"],
                       "views": [100.0, 120.0, 90.0, 40.0, 50.0]})
    a = _a(ax.attach_wikipedia(df, pv, known_through="2024-12-31"))
    assert np.isnan(a.loc["2024-01-02", "wiki_views"])                 # before the first usable day
    assert a.loc["2024-01-03", "wiki_views"] == 100 and a.loc["2024-01-04", "wiki_views"] == 120
    assert np.isnan(a.loc["2024-01-05", "wiki_views"])                 # Thu 01-04 had no row: missing, not zero
    assert a.loc["2024-01-08", "wiki_views"] == 90 + 40 + 50           # Fri + Sat + Sun -> Monday
    assert _a(ax.attach_wikipedia(df, pv, known_through="2024-12-31"), "BBB")["wiki_views"].isna().all()
    a2 = _a(ax.attach_wikipedia(df, pv))                                # known only through 2024-01-07
    assert a2.loc["2024-01-08", "wiki_views"] == 180 and a2.loc["2024-01-09":, "wiki_views"].isna().all()


def test_future_days_never_move_the_past_and_dsl_sees_it():
    df = _panel()
    base = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-02"], "views": [100.0]})
    fut = pd.concat([base, pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-20"], "views": [9999.0]})])
    a, a2 = _a(ax.attach_wikipedia(df, base, known_through="2024-12-31")), _a(ax.attach_wikipedia(df, fut, known_through="2024-12-31"))
    cut = pd.Timestamp("2024-01-19")
    np.testing.assert_array_equal(a.loc[:cut, "wiki_views"].to_numpy(), a2.loc[:cut, "wiki_views"].to_numpy())
    out = ax.attach(df, **{k: NONE for k in ax.SOURCE_KWARGS}, wiki=base)
    assert "wiki_views" in out.columns and "wiki_views" in dsl.describe_ops() and "wiki_views" in dsl.AUX_FIELDS
    s = dsl.compile_expression("log(fillna(wiki_views, 0))", out)
    assert s.notna().sum() > 0
