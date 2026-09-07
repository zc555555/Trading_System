"""NYSE trading calendar: weekdays minus exchange holidays, dependency-free.

Until 2026-09 this module was weekdays-only ("holidays only shift dates by
1-2 days"). The 2026-09 review pointed out what that costs: on a holiday
the nightly run generated signals from stale data and opened a second
tranche on the same session's data, and scheduled close dates landed on
days the market was shut. The rules below are the published NYSE holiday
rules (observed-day shifts included) plus the special closures, so
`is_trading_day` is exact for the years this system runs in. The
orchestrator additionally cross-checks against the broker's calendar
each run and logs a FAILURES line on any disagreement.

Public API (unchanged names): DateLike, _to_date, is_trading_day,
add_trading_days, trading_days_between, today; new: nyse_holidays,
previous_trading_day, next_trading_day, last_session,
last_completed_session, eastern_now.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Dict, FrozenSet, Optional, Union

import numpy as np

DateLike = Union[str, date, datetime, np.datetime64]

# unscheduled full-day closures (national days of mourning, Sandy)
SPECIAL_CLOSURES: FrozenSet[date] = frozenset({
    date(2012, 10, 29), date(2012, 10, 30),      # Hurricane Sandy
    date(2018, 12, 5),                           # G.H.W. Bush
    date(2025, 1, 9),                            # J. Carter
})

MARKET_CLOSE = time(16, 0)


def _to_date(d: DateLike) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, np.datetime64):
        return _to_date(str(np.datetime_as_string(d, unit="D")))
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


# ---------------------------------------------------------------------------
# holiday rules
# ---------------------------------------------------------------------------
def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7          # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + 1, 1, 1) - timedelta(days=1) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> Optional[date]:
    """Saturday holidays are observed on Friday, Sunday ones on Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


_HOLIDAY_CACHE: Dict[int, FrozenSet[date]] = {}


def nyse_holidays(year: int) -> FrozenSet[date]:
    """Full-day NYSE closures in `year` (weekday dates only)."""
    if year in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[year]
    h = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 6:                         # Sunday -> Monday
        h.add(ny + timedelta(days=1))
    elif ny.weekday() < 5:                        # a Saturday New Year is NOT observed on the Friday
        h.add(ny)
    h.add(_nth_weekday(year, 1, 0, 3))            # Martin Luther King Jr. Day
    h.add(_nth_weekday(year, 2, 0, 3))            # Presidents' Day
    h.add(_easter(year) - timedelta(days=2))      # Good Friday
    h.add(_last_weekday(year, 5, 0))              # Memorial Day
    if year >= 2022:
        h.add(_observed(date(year, 6, 19)))       # Juneteenth
    h.add(_observed(date(year, 7, 4)))            # Independence Day
    h.add(_nth_weekday(year, 9, 0, 1))            # Labor Day
    h.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving
    h.add(_observed(date(year, 12, 25)))          # Christmas
    h |= {d for d in SPECIAL_CLOSURES if d.year == year}
    out = frozenset(d for d in h if d is not None and d.weekday() < 5)
    _HOLIDAY_CACHE[year] = out
    return out


# ---------------------------------------------------------------------------
# day arithmetic
# ---------------------------------------------------------------------------
def is_trading_day(d: DateLike) -> bool:
    dd = _to_date(d)
    return dd.weekday() < 5 and dd not in nyse_holidays(dd.year)


def add_trading_days(d: DateLike, n: int) -> date:
    """Return ``d`` plus ``n`` trading days.

    Handles n >= 0 and n < 0. If ``d`` itself is not a trading day, advances
    (in the direction of n) to the next trading day first, so
    ``add_trading_days(Sat, 0) == next Mon``.
    """
    start = _to_date(d)
    direction = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = start
    while not is_trading_day(cur):
        cur = cur + timedelta(days=direction)
    while remaining > 0:
        cur = cur + timedelta(days=direction)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def previous_trading_day(d: DateLike) -> date:
    cur = _to_date(d) - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def next_trading_day(d: DateLike) -> date:
    cur = _to_date(d) + timedelta(days=1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def last_session(as_of: DateLike) -> date:
    """The most recent trading day on or before `as_of`."""
    dd = _to_date(as_of)
    return dd if is_trading_day(dd) else previous_trading_day(dd)


def trading_days_between(start: DateLike, end: DateLike) -> int:
    """Inclusive count of trading days between start and end."""
    s = _to_date(start)
    e = _to_date(end)
    if s > e:
        s, e = e, s
    days = (e - s).days + 1
    return sum(1 for i in range(days) if is_trading_day(s + timedelta(days=i)))


def today() -> date:
    return date.today()


# ---------------------------------------------------------------------------
# exchange clock
# ---------------------------------------------------------------------------
class _USEastern(tzinfo):
    """US/Eastern with the post-2007 DST rule (2nd Sunday of March 02:00 to
    1st Sunday of November 02:00). Fallback when the zoneinfo database is
    missing (Windows without tzdata)."""
    STD = timedelta(hours=-5)
    DST = timedelta(hours=1)
    ZERO = timedelta(0)

    def _dst_window(self, year: int):
        start = _nth_weekday(year, 3, 6, 2)
        end = _nth_weekday(year, 11, 6, 1)
        return datetime(start.year, start.month, start.day, 2), datetime(end.year, end.month, end.day, 2)

    def dst(self, dt):
        if dt is None:
            return self.ZERO
        start, end = self._dst_window(dt.year)
        naive = dt.replace(tzinfo=None)
        return self.DST if start <= naive < end else self.ZERO

    def utcoffset(self, dt):
        return self.STD + self.dst(dt)

    def tzname(self, dt):
        return "EDT" if self.dst(dt) else "EST"

    def fromutc(self, dt):
        std = dt.replace(tzinfo=None) + self.STD
        start, end = self._dst_window(std.year)
        # DST boundaries expressed in standard local time
        if start <= std < end - self.DST:
            return (std + self.DST).replace(tzinfo=self)
        return std.replace(tzinfo=self)


def _eastern_tz():
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        datetime(2026, 1, 1, tzinfo=timezone.utc).astimezone(tz)   # raises if tzdata is missing
        return tz
    except Exception:                                # noqa: BLE001
        return _USEastern()


EASTERN = _eastern_tz()


def eastern_now(now: Optional[datetime] = None) -> datetime:
    """Current exchange-local time. `now` may be any aware datetime (a naive
    one is taken as the machine's local time)."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    return now.astimezone(EASTERN)


def last_completed_session(now_et: datetime) -> date:
    """The last session whose close (16:00 ET) has already happened -- the
    only data date a freshly generated signal file may legitimately carry."""
    d = now_et.date()
    if is_trading_day(d) and now_et.time() >= MARKET_CLOSE:
        return d
    return previous_trading_day(d)
