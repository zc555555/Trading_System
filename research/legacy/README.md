# Legacy / archived

Files in this folder are **no longer wired into the production pipeline**.
They are kept for historical reference only.

| File | Superseded by | Notes |
|---|---|---|
| `get_daily_signals_old.py` | `../get_daily_signals_multi_factor.py` | Original Plan-A signal generator |
| `get_daily_signals_weighted.py` | `../get_daily_signals_multi_factor.py` | 5-day weighted variant |
| `build_output.log` | — | Captured build log from a past run |
| `test_output.log` | — | Captured test log from a past run |

If you find yourself reaching for one of these, double-check that the
current pipeline (`get_daily_signals_multi_factor.py`) really doesn't
already cover the use case before resurrecting it.
