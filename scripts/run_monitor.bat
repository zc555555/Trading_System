@echo off
REM ===========================================================================
REM Intraday risk monitor (stop-loss / take-profit / EOD flat).
REM Designed to be triggered by Windows Task Scheduler every 5 minutes
REM during market hours (14:30 - 21:05 local time).
REM ===========================================================================
chcp 65001 >nul
cd /d "%~dp0\.."

echo ================================================================================
echo Dynamic Trading Monitor
echo ================================================================================
echo   - Per-position stop-loss:    -2.5%%
echo   - Per-position take-profit:  +2.5%%
echo   - Account daily loss limit:  -3.0%%
echo   - End-of-day flat at 16:00 ET
echo.
echo Press Ctrl+C to stop.
echo ================================================================================
echo.

python monitor_dynamic_trading.py --auto
pause
