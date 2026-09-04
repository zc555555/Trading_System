"""Pins the Form 4 insider fields and the FINRA short-interest fields
(research/mining/aux_fields.attach_form4 / attach_short_interest): a filing
is usable from the first session strictly after its filing day, weekend
filings land on Monday, counts are 0 for covered-but-quiet symbols and NaN
for uncovered ones, net fraction uses the EDGAR share count, a short-interest
figure becomes public SHORT_INTEREST_LAG sessions after settlement, expires
after SHORT_MAX_AGE sessions, and future data never moves the past."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=80):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")   # Mon 2024-01-01 ...
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


FILINGS = pd.DataFrame({"symbol": ["AAA", "BBB"], "filed": ["2023-12-20", "2023-12-20"], "shares_out": [1_000_000.0, 2_000_000.0]})


def _attach(df, **kw):
    return ax.attach(df, source=NONE, filings=FILINGS, news_source=NONE, news=None,
                     form4_source=NONE, short_source=NONE, **kw)


def _form4(df, ev, known_through="2024-12-31"):
    return ax.attach_form4(ax.attach_filings(df, FILINGS), ev, known_through=known_through)


def _a(out, sym="AAA"):
    sub = out[out.symbol == sym]
    return sub.set_index(sub["date"].dt.tz_localize(None))


def test_form4_next_session_weekend_counts_and_net_fraction():
    df = _panel()
    ev = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA"],
                       "filed": ["2024-01-02", "2024-01-06", "2024-01-06"],          # Tue, Sat, Sat
                       "buy_shares": [1000.0, 0.0, 500.0], "sell_shares": [0.0, 3000.0, 0.0],
                       "n_buyers": [1.0, 0.0, 2.0], "n_sellers": [0.0, 1.0, 0.0]})
    a = _a(_attach(df, form4=ev))                      # known through the last filing (2024-01-06 -> Mon 01-08)
    assert a.loc["2024-01-09", "insider_buys"] != a.loc["2024-01-09", "insider_buys"]   # after the source ends: unknown
    a = _a(_form4(df, ev))                             # same events, source declared known through year-end
    assert a.loc["2024-01-02", "insider_buys"] == 0 and a.loc["2024-01-03", "insider_buys"] == 1
    assert abs(a.loc["2024-01-03", "insider_net_frac"] - 1000 / 1_000_000) < 1e-12
    # Saturday filings land on Monday 2024-01-08, aggregated
    assert a.loc["2024-01-08", "insider_buys"] == 2 and a.loc["2024-01-08", "insider_sells"] == 1
    assert abs(a.loc["2024-01-08", "insider_net_frac"] - (500 - 3000) / 1_000_000) < 1e-12
    assert a.loc["2024-01-09", "insider_buys"] == 0 and a.loc["2024-01-09", "insider_net_frac"] == 0
    b = _a(_form4(df, ev), "BBB")
    assert b[["insider_buys", "insider_sells", "insider_net_frac"]].isna().all().all()   # uncovered


def test_form4_future_filing_never_moves_the_past():
    df = _panel()
    base = pd.DataFrame({"symbol": ["AAA"], "filed": ["2024-01-02"], "buy_shares": [100.0], "sell_shares": [0.0], "n_buyers": [1.0], "n_sellers": [0.0]})
    fut = pd.concat([base, pd.DataFrame({"symbol": ["AAA"], "filed": ["2024-02-20"], "buy_shares": [0.0], "sell_shares": [9e6], "n_buyers": [0.0], "n_sellers": [5.0]})])
    a, a2 = _a(_form4(df, base)), _a(_form4(df, fut))
    cut = pd.Timestamp("2024-02-20")
    for c in ("insider_buys", "insider_sells", "insider_net_frac"):
        np.testing.assert_array_equal(a.loc[:cut, c].to_numpy(), a2.loc[:cut, c].to_numpy())


def test_short_interest_lag_ratio_and_expiry():
    df = _panel(n_days=80)
    si = pd.DataFrame({"symbol": ["AAA"], "settlement_date": ["2024-01-12"],            # Friday, session ordinal 9
                       "short_interest": [50_000.0], "avg_daily_volume": [10_000.0], "days_to_cover": [5.0]})
    a = _a(_attach(df, short=si))
    sessions = list(a.index)
    eff = sessions[9 + ax.SHORT_INTEREST_LAG]
    before = sessions[9 + ax.SHORT_INTEREST_LAG - 1]
    assert np.isnan(a.loc[before, "short_ratio"]) and np.isnan(a.loc[before, "days_to_cover"])
    assert abs(a.loc[eff, "short_ratio"] - 0.05) < 1e-12 and a.loc[eff, "days_to_cover"] == 5.0
    last_ok = sessions[9 + ax.SHORT_INTEREST_LAG + ax.SHORT_MAX_AGE]
    assert abs(a.loc[last_ok, "short_ratio"] - 0.05) < 1e-12
    assert np.isnan(a.loc[sessions[9 + ax.SHORT_INTEREST_LAG + ax.SHORT_MAX_AGE + 1], "short_ratio"])
    b = _a(_attach(df, short=si), "BBB")
    assert b["short_ratio"].isna().all()


def test_short_interest_future_figure_never_moves_the_past_and_dsl_sees_fields():
    df = _panel(n_days=80)
    base = pd.DataFrame({"symbol": ["AAA"], "settlement_date": ["2024-01-12"], "short_interest": [50_000.0], "avg_daily_volume": [1e4], "days_to_cover": [5.0]})
    fut = pd.concat([base, pd.DataFrame({"symbol": ["AAA"], "settlement_date": ["2024-02-15"], "short_interest": [9e5], "avg_daily_volume": [1e4], "days_to_cover": [90.0]})])
    a, a2 = _a(_attach(df, short=base)), _a(_attach(df, short=fut))
    sessions = list(a.index)
    cut = sessions[sessions.index(pd.Timestamp("2024-02-15")) + ax.SHORT_INTEREST_LAG - 1]
    np.testing.assert_array_equal(a.loc[:cut, "short_ratio"].to_numpy(), a2.loc[:cut, "short_ratio"].to_numpy())
    out = _attach(df, short=base)
    s = dsl.compile_expression("delta(short_ratio, 5) + days_to_cover", out)
    assert s.notna().sum() > 0
    for f in ("insider_buys", "insider_sells", "insider_net_frac", "short_ratio", "days_to_cover"):
        assert f in dsl.AUX_FIELDS and f in dsl.describe_ops()
    try:
        dsl.compile_expression("rank(insider_buys)", out)
    except dsl.DSLError as e:
        assert "not available" in str(e)
    else:
        raise AssertionError("insider fields must be refused when their source is absent")
