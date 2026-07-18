@echo off
REM ===========================================================================
REM Daily auto-trading pipeline:
REM   fetch data -> generate signals -> flatten -> place new orders -> report
REM Recommended schedule: 21:00 ET, weekdays.
REM ===========================================================================
chcp 65001 >nul
cd /d "%~dp0\.."
python run_auto_trading.py
pause
