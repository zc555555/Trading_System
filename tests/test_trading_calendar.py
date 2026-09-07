"""NYSE calendar pins (2026-09 review: holidays were ignored)."""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from trading import trading_calendar as cal  # noqa: E402


def test_2026_holidays_and_observed_rules():
    closed = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),   # NY, MLK, Presidents, Good Friday
              date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),                    # Memorial, Juneteenth, July 4 observed (Sat -> Fri)
              date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}                  # Labor Day, Thanksgiving, Christmas
    assert cal.nyse_holidays(2026) == frozenset(closed)
    for d in closed:
        assert not cal.is_trading_day(d)
    assert cal.is_trading_day(date(2026, 9, 8)) and cal.is_trading_day(date(2026, 9, 4))
    assert not cal.is_trading_day(date(2026, 9, 5))                                      # Saturday
    # Christmas 2027 is a Saturday -> Friday 24th closed; New Year 2028 is a Saturday -> NOT observed
    assert date(2027, 12, 24) in cal.nyse_holidays(2027)
    assert date(2027, 12, 31) not in cal.nyse_holidays(2027) and cal.is_trading_day(date(2027, 12, 31))
    assert date(2028, 1, 3) not in cal.nyse_holidays(2028)
    # Sunday holiday -> Monday
    assert date(2023, 1, 2) in cal.nyse_holidays(2023)
    # special closures
    assert not cal.is_trading_day(date(2025, 1, 9))


def test_day_arithmetic_skips_holidays():
    assert cal.add_trading_days(date(2026, 9, 4), 1) == date(2026, 9, 8)
    assert cal.add_trading_days(date(2026, 9, 4), 20) == date(2026, 10, 5)             # Labor Day inside the window
    assert cal.add_trading_days(date(2026, 9, 5), 0) == date(2026, 9, 8)               # weekend then holiday
    assert cal.previous_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)
    assert cal.next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)
    assert cal.last_session(date(2026, 9, 7)) == date(2026, 9, 4)
    assert cal.trading_days_between(date(2026, 9, 4), date(2026, 9, 8)) == 2


def test_exchange_clock_and_last_completed_session():
    edt = cal.eastern_now(datetime(2026, 9, 7, 20, 0, tzinfo=timezone.utc))
    est = cal.eastern_now(datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc))
    assert (edt.hour, edt.minute) == (16, 0) and est.hour == 15
    # Labor Day evening: the last completed session is the Friday before
    assert cal.last_completed_session(edt) == date(2026, 9, 4)
    # Tuesday before the close -> Friday; after the close -> Tuesday
    before = datetime(2026, 9, 8, 15, 59, tzinfo=cal.EASTERN)
    after = datetime(2026, 9, 8, 16, 0, tzinfo=cal.EASTERN)
    assert cal.last_completed_session(before) == date(2026, 9, 4)
    assert cal.last_completed_session(after) == date(2026, 9, 8)


def test_fallback_eastern_tz_matches_dst_rule():
    tz = cal._USEastern()
    # 2026 DST: Mar 8 - Nov 1
    assert datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc).astimezone(tz).hour == 1     # still EST
    assert datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc).astimezone(tz).hour == 3      # jumps to EDT
    assert datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc).astimezone(tz).hour == 1    # EDT
    assert datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc).astimezone(tz).hour == 1     # EST again
    assert datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc).astimezone(tz).tzname() == "EDT"
