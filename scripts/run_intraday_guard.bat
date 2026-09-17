@echo off
rem Intraday daily-loss guard tick (Task Scheduler StockPredict_IntradayGuard,
rem every 5 min 14:00-22:00 UK on weekdays; the script exits at once outside
rem the US session, on a closed day, or when the account is already halted).
cd /d "%~dp0.."
"research\venv\Scripts\python.exe" trading\intraday_guard.py >> "trading_logs\intraday_guard_run.log" 2>&1
