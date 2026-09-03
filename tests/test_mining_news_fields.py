"""Pins the GDELT news fields (research/mining/aux_fields.attach_news): a
calendar day's news is usable from the first session strictly after it,
weekend news lands on Monday, tone is article-weighted, uncovered symbols
are NaN while covered symbols with no article are 0, and future news never
changes past values."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "research"))

from mining import aux_fields as ax, dsl  # noqa: E402

NONE = Path("nonexistent")


def _panel(n_days=30):
    dates = pd.bdate_range("2024-01-01", periods=n_days, tz="America/New_York")   # Mon 2024-01-01 ...
    rows = [pd.DataFrame({"date": dates, "symbol": s, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
            for s in ("AAA", "BBB")]
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _a(out, sym="AAA"):
    return out[out.symbol == sym].set_index(out[out.symbol == sym]["date"].dt.tz_localize(None))


def test_news_lands_on_the_next_session_and_weekends_on_monday():
    df = _panel()
    news = pd.DataFrame({"symbol": ["AAA"] * 4,
                         "date": ["2024-01-02", "2024-01-05", "2024-01-06", "2024-01-07"],   # Tue, Fri, Sat, Sun
                         "gdelt_tone": [2.0, -1.0, 3.0, 5.0], "gdelt_articles": [10, 10, 10, 30]})
    a = _a(ax.attach(df, source=NONE, news=news))
    assert np.isnan(a.loc["2024-01-02", "news_tone"]) and a.loc["2024-01-02", "news_articles"] == 0   # same day: not yet
    assert a.loc["2024-01-03", "news_tone"] == 2.0 and a.loc["2024-01-03", "news_articles"] == 10
    # Fri + Sat + Sun -> Monday 2024-01-08, article-weighted: (-10 + 30 + 150) / 50 = 3.4
    assert abs(a.loc["2024-01-08", "news_tone"] - 3.4) < 1e-12 and a.loc["2024-01-08", "news_articles"] == 50
    assert a.loc["2024-01-09", "news_articles"] == 0 and np.isnan(a.loc["2024-01-09", "news_tone"])
    b = _a(ax.attach(df, source=NONE, news=news), "BBB")
    assert b["news_tone"].isna().all() and b["news_articles"].isna().all()       # not covered at all


def test_future_news_never_changes_past_values_and_dsl_uses_the_fields():
    df = _panel()
    base = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-02"], "gdelt_tone": [2.0], "gdelt_articles": [5]})
    fut = pd.concat([base, pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-20"], "gdelt_tone": [-9.0], "gdelt_articles": [99]})])
    a, a2 = _a(ax.attach(df, source=NONE, news=base)), _a(ax.attach(df, source=NONE, news=fut))
    cut = pd.Timestamp("2024-01-20")
    for c in ("news_tone", "news_articles"):
        np.testing.assert_array_equal(a.loc[:cut, c].to_numpy(), a2.loc[:cut, c].to_numpy())
    out = ax.attach(df, source=NONE, news=base)
    s = dsl.compile_expression("ts_sum(news_articles, 5)", out)
    assert s.notna().sum() > 0            # one symbol, so no cross-sectional rank here
    assert "news_tone" in dsl.describe_ops()
    try:
        dsl.compile_expression("rank(news_tone)", df)
    except dsl.DSLError as e:
        assert "not available" in str(e)
    else:
        raise AssertionError("a panel without the news source must refuse news fields")
