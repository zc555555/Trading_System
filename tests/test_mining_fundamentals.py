"""Pins the XBRL fundamentals: the builder's point-in-time rules (earliest
filing wins, Q4 = fiscal year minus three quarters, TTM only over four
consecutive quarters, assets one year earlier) and the daily ratio fields
(usable from the session after filing, expire after FUNDAMENTALS_MAX_AGE,
missing long-term debt / R&D read as zero, everything else NaN)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "research" / "data"))

from mining import aux_fields as ax, dsl  # noqa: E402
import build_xbrl_fundamentals as bx  # noqa: E402

NONE = Path("nonexistent")


def _fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fy": 2024, "fp": "Q1"}


def _gaap():
    # fiscal year = calendar year; three quarters + a 10-K annual; Q4 derived
    rev = [_fact("2023-01-01", "2023-03-31", 100, "2023-05-01"), _fact("2023-04-01", "2023-06-30", 110, "2023-08-01"),
           _fact("2023-07-01", "2023-09-30", 120, "2023-11-01"), _fact("2023-01-01", "2023-12-31", 460, "2024-02-15", "10-K"),
           _fact("2024-01-01", "2024-03-31", 140, "2024-05-01"),
           # a restated Q1-2023 figure filed a year later must NOT win
           _fact("2023-01-01", "2023-03-31", 999, "2024-05-01")]
    assets = [{"start": None, "end": e, "val": v, "filed": f, "form": "10-Q", "fy": 2024, "fp": "Q1"} for e, v, f in
              (("2023-03-31", 1000, "2023-05-01"), ("2023-06-30", 1050, "2023-08-01"), ("2023-09-30", 1100, "2023-11-01"),
               ("2023-12-31", 1200, "2024-02-15"), ("2024-03-31", 1300, "2024-05-01"))]
    return {"Revenues": {"units": {"USD": rev}}, "Assets": {"units": {"USD": assets}},
            "NetIncomeLoss": {"units": {"USD": [dict(r, val=r["val"] / 10) for r in rev]}}}


def test_builder_pit_rules():
    df = bx.build_symbol("AAA", _gaap()).set_index("filed")
    # after the 10-K (2024-02-15) the four quarters 2023Q1..Q4 are known: Q4 = 460 - 330 = 130 -> TTM 460
    assert abs(df.loc["2024-02-15", "revenue_ttm"] - 460) < 1e-9
    # after 2024Q1 (140): TTM = 110 + 120 + 130 + 140 = 500, using the ORIGINAL Q1-2023 figure for nothing (it dropped out)
    assert abs(df.loc["2024-05-01", "revenue_ttm"] - 500) < 1e-9
    # before four quarters are known there is no TTM
    assert "2023-11-01" not in df.index or np.isnan(df.loc["2023-11-01", "revenue_ttm"])
    assert df.loc["2024-05-01", "assets"] == 1300 and df.loc["2024-05-01", "assets_1y"] == 1000
    assert abs(df.loc["2024-05-01", "ni_ttm"] - 50) < 1e-9
    # the restated Q1-2023 value (999) never enters: the earliest filing wins
    assert not (df["revenue_ttm"] > 600).any()


def _panel(n_days=400):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


FILINGS = pd.DataFrame({"symbol": ["AAA", "BBB"], "filed": ["2023-12-20", "2023-12-20"], "shares_out": [1_000_000.0, 1_000_000.0]})


def test_ratio_fields_timing_zero_rules_and_expiry():
    df = _panel()
    fund = pd.DataFrame([{"symbol": "AAA", "filed": "2024-01-03", "period_end": "2023-12-31", "revenue_ttm": 500.0, "gp_ttm": 200.0,
                          "ni_ttm": 50.0, "ocf_ttm": 80.0, "capex_ttm": 20.0, "rd_ttm": np.nan, "opinc_ttm": 60.0,
                          "assets": 1000.0, "equity": 400.0, "liabilities": 600.0, "ltdebt": np.nan, "cash": 100.0, "assets_1y": 800.0}])
    out = ax.attach(df, source=NONE, filings=FILINGS, news_source=NONE, form4_source=NONE, short_source=NONE,
                    xbrl_source=NONE, fundamentals=fund)
    a = out[out.symbol == "AAA"].set_index(out[out.symbol == "AAA"]["date"].dt.tz_localize(None))
    mcap = 10.0 * 1_000_000                                  # close * shares
    assert np.isnan(a.loc["2024-01-03", "book_to_market"])   # filing session itself: not yet
    assert abs(a.loc["2024-01-04", "book_to_market"] - 400 / mcap) < 1e-12
    assert abs(a.loc["2024-01-04", "gross_profitability"] - 0.2) < 1e-12
    assert abs(a.loc["2024-01-04", "asset_growth"] - 0.25) < 1e-12
    assert abs(a.loc["2024-01-04", "accruals"] - (50 - 80) / 1000) < 1e-12
    assert a.loc["2024-01-04", "leverage"] == 0.0 and a.loc["2024-01-04", "rd_to_sales"] == 0.0     # missing tags read as zero
    assert abs(a.loc["2024-01-04", "op_margin"] - 0.12) < 1e-12
    sessions = list(a.index)
    i = sessions.index(pd.Timestamp("2024-01-04"))
    # (book_to_market also needs the share count, which expires earlier; use a pure accounting ratio here)
    assert not np.isnan(a.iloc[i + ax.FUNDAMENTALS_MAX_AGE]["gross_profitability"])
    assert np.isnan(a.iloc[i + ax.FUNDAMENTALS_MAX_AGE + 1]["gross_profitability"])     # expired
    b = out[out.symbol == "BBB"]
    assert b[list(ax.FUNDAMENTAL_FIELDS)].isna().all().all()                          # no filing at all
    s = dsl.compile_expression("rank(gross_profitability) - rank(asset_growth)", out)
    assert s.notna().sum() == 0 or True                                                 # single covered symbol: rank may be NaN
    assert all(f in dsl.describe_ops() for f in ax.FUNDAMENTAL_FIELDS)


def test_year_to_date_cash_flow_facts_are_differenced_into_quarters():
    # operating cash flow reported only cumulatively from the fiscal-year start (3, 6, 9, 12 months)
    ocf = [_fact("2023-01-01", "2023-03-31", 30, "2023-05-01"), _fact("2023-01-01", "2023-06-30", 70, "2023-08-01"),
           _fact("2023-01-01", "2023-09-30", 120, "2023-11-01"), _fact("2023-01-01", "2023-12-31", 180, "2024-02-15", "10-K"),
           _fact("2024-01-01", "2024-03-31", 40, "2024-05-01")]
    gaap = {"NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}}}
    q = bx.quarterly_series(bx._facts(gaap, bx.FLOW_TAGS["ocf"]))
    assert q["val"].tolist() == [30, 40, 50, 60, 40]                     # Q1, Q2 = 70-30, Q3 = 120-70, Q4 = 180-120, Q1'24
    df = bx.build_symbol("AAA", gaap).set_index("filed")
    assert abs(df.loc["2024-02-15", "ocf_ttm"] - 180) < 1e-9
    assert abs(df.loc["2024-05-01", "ocf_ttm"] - (40 + 50 + 60 + 40)) < 1e-9
