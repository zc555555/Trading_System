"""Pins the auxiliary DSL fields (research/mining/aux_fields.py, EDGAR-based):
share counts join as-of the filing session + 1, market cap and turnover
derive from them, filing_days counts sessions since the last usable filing,
a stale count expires, future filings never change past values, and the DSL
refuses a field whose source is absent."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=300):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")
    rows = []
    for s in ("AAA", "BBB"):
        rows.append(pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0,
                                  "close": 10.0, "volume": 1000.0}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _sessions(df):
    return df["date"].dt.tz_localize(None).drop_duplicates().sort_values().reset_index(drop=True)


def _a(out, df):
    return out[out.symbol == "AAA"].set_index(df[df.symbol == "AAA"]["date"].dt.tz_localize(None))


def test_filing_join_marketcap_turnover_and_filing_days():
    df = _panel()
    s = _sessions(df)
    filings = pd.DataFrame({"symbol": ["AAA", "AAA"], "filed": [s[5].strftime("%Y-%m-%d"), s[70].strftime("%Y-%m-%d")],
                            "shares_out": [2e6, 4e6]})
    out = ax.attach(df, source=NONE, filings=filings)
    a = _a(out, df)
    assert np.isnan(a.loc[s[5], "marketcap"]) and np.isnan(a.loc[s[5], "filing_days"])   # filing session itself
    assert a.loc[s[6], "filing_days"] == 0 and a.loc[s[6], "marketcap"] == 10.0 * 2e6 / 1e6
    assert abs(a.loc[s[6], "turnover"] - 1000.0 / 2e6) < 1e-15
    assert a.loc[s[70], "marketcap"] == 20.0 and a.loc[s[71], "marketcap"] == 40.0      # new count from s[71]
    assert a.loc[s[71], "filing_days"] == 0 and a.loc[s[90], "filing_days"] == 19
    assert out[out.symbol == "BBB"][["marketcap", "turnover", "filing_days"]].isna().all().all()


def test_stale_share_count_expires_and_weekend_filing_maps_forward():
    df = _panel()
    s = _sessions(df)
    filings = pd.DataFrame({"symbol": ["AAA"], "filed": [s[5].strftime("%Y-%m-%d")], "shares_out": [2e6]})
    a = _a(ax.attach(df, source=NONE, filings=filings), df)
    last_ok = s[6 + ax.SHARES_MAX_AGE]
    assert a.loc[last_ok, "marketcap"] == 20.0 and np.isnan(a.loc[s[7 + ax.SHARES_MAX_AGE], "marketcap"])
    assert a.loc[last_ok, "filing_days"] == ax.SHARES_MAX_AGE
    assert np.isnan(a.loc[s[7 + ax.SHARES_MAX_AGE], "filing_days"])   # a company that stopped filing: unknown, not stale
    i_fri = next(i for i in range(8, 30) if s[i].weekday() == 4)
    sat = s[i_fri] + pd.Timedelta(days=1)
    assert sat.weekday() == 5
    a2 = _a(ax.attach(df, source=NONE, filings=pd.DataFrame({"symbol": ["AAA"], "filed": [sat.strftime("%Y-%m-%d")],
                                                              "shares_out": [3e6]})), df)
    assert np.isnan(a2.loc[s[i_fri + 1], "filing_days"])        # Monday: the filing's own session
    assert a2.loc[s[i_fri + 2], "filing_days"] == 0              # Tuesday: first usable


def test_future_filings_never_change_past_values():
    df = _panel()
    s = _sessions(df)
    base = pd.DataFrame({"symbol": ["AAA"], "filed": [s[5].strftime("%Y-%m-%d")], "shares_out": [2e6]})
    fut = pd.DataFrame({"symbol": ["AAA", "AAA"], "filed": [s[5].strftime("%Y-%m-%d"), s[100].strftime("%Y-%m-%d")],
                        "shares_out": [2e6, 9e6]})
    a, a3 = _a(ax.attach(df, source=NONE, filings=base), df), _a(ax.attach(df, source=NONE, filings=fut), df)
    for c in ("marketcap", "turnover", "filing_days"):
        np.testing.assert_array_equal(a3.loc[:s[100], c].to_numpy(), a.loc[:s[100], c].to_numpy())
    assert a3.loc[s[101], "filing_days"] == 0 and a3.loc[s[101], "marketcap"] == 90.0


def test_dsl_refuses_an_unavailable_field_and_uses_an_available_one():
    df = _panel()
    with pytest.raises(dsl.DSLError, match="not available"):
        dsl.compile_expression("rank(filing_days)", df)
    with pytest.raises(dsl.DSLError, match="not available"):
        dsl.compile_expression("log(marketcap)", df)
    filings = pd.DataFrame({"symbol": ["AAA"], "filed": [_sessions(df)[0].strftime("%Y-%m-%d")], "shares_out": [1e6]})
    out = ax.attach(df, source=NONE, filings=filings)
    assert dsl.compile_expression("ts_max(filing_days, 3) + log(marketcap) + rank(turnover)", out).notna().sum() > 0
    assert set(dsl.AUX_FIELDS) <= set(dsl.FIELDS) and "filing_days" in dsl.describe_ops()
    assert dsl.lookback(dsl.parse("filing_days")) == 0
    assert ax.attach(df, source=NONE, news_source=NONE, form4_source=NONE, short_source=NONE) is df   # no source at all
