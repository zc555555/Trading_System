@echo off
rem Unattended nightly trading entry (Task Scheduler: Mon-Fri 21:00 UK).
rem Uses the research venv interpreter and appends to a daily log so a
rem failed overnight run is diagnosable the next morning. Failures also
rem land in trading_logs\FAILURES.log and raise a desktop notice --
rem the 2026-08-08 crash sat unnoticed in the daily log for nine days.
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
echo ================ run started %date% %time% ================ >> "trading_logs\auto_run_%DT%.log"
"research\venv\Scripts\python.exe" run_auto_trading.py >> "trading_logs\auto_run_%DT%.log" 2>&1
set RC=%errorlevel%
echo ================ run finished %date% %time% exit=%RC% ================ >> "trading_logs\auto_run_%DT%.log"
if not "%RC%"=="0" (
    echo %date% %time% NIGHTLY TRADING RUN FAILED exit=%RC% see auto_run_%DT%.log >> "trading_logs\FAILURES.log"
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0notify_failure.ps1" "StockPredict 夜间交易失败" "exit=%RC%, 详见 trading_logs\auto_run_%DT%.log"
)
rem Daily live-book attribution (descriptive analytics; places no orders).
"research\venv\Scripts\python.exe" log_attribution.py >> "trading_logs\auto_run_%DT%.log" 2>&1
if not "%errorlevel%"=="0" (
    echo %date% %time% ATTRIBUTION LOG FAILED see auto_run_%DT%.log >> "trading_logs\FAILURES.log"
)
rem Daily fill capture for the execution A/B (was Sunday-only; two missed
rem Sundays left the fill record stale at 2026-08-14). Non-fatal.
"research\venv\Scripts\python.exe" research\evaluation\experiments\slippage_calibration.py >> "trading_logs\auto_run_%DT%.log" 2>&1
if not "%errorlevel%"=="0" (
    echo %date% %time% SLIPPAGE CAPTURE FAILED see auto_run_%DT%.log >> "trading_logs\FAILURES.log"
)
