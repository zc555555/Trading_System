"""US trading-day arithmetic.

Pandas's BusinessDay handles Mon-Fri but not US-specific holidays. For our
purposes ('was today a trading day?', 'how many trading days between A and B?')
weekdays-only is close enough -- holidays only shift dates by 1-2 days a few
times a year, and the staggered orchestrator already tolerates being 'off by
one' because it queries the registry against today's date, not a precomputed
schedule.

If you need exact NYSE handling later, swap to ``pandas_market_calendars``
(requires ``pip install pandas-market-calendars``) and replace the impls
here. The public API will not change.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Union

import numpy as np

DateLike = Union[str, date, datetime, np.datetime64]


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


def is_trading_day(d: DateLike) -> bool:
    """Approximate: Mon-Fri only. Does not exclude US holidays."""
    return _to_date(d).weekday() < 5


def add_trading_days(d: DateLike, n: int) -> date:
    """Return ``d`` plus ``n`` trading days (weekdays only).

    Handles n >= 0 and n < 0. If ``d`` itself is a weekend, advances to the
    next weekday first (so ``add_trading_days(Sat, 0) == next Mon``).
    """
    start = _to_date(d)
    direction = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = start

    # Snap to a weekday if the start lands on a weekend
    while not is_trading_day(cur):
        cur = cur + timedelta(days=direction)

    while remaining > 0:
        cur = cur + timedelta(days=direction)
        if is_trading_day(cur):
            remaining -= 1
    return cur


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
