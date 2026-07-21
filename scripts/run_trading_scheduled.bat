@echo off
rem Unattended nightly trading entry (Task Scheduler: Mon-Fri 21:00 UK).
rem Uses the research venv interpreter and appends to a daily log so a
rem failed overnight run is diagnosable the next morning.
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
echo ================ run started %date% %time% ================ >> "trading_logs\auto_run_%DT%.log"
"research\venv\Scripts\python.exe" run_auto_trading.py >> "trading_logs\auto_run_%DT%.log" 2>&1
echo ================ run finished %date% %time% exit=%errorlevel% ================ >> "trading_logs\auto_run_%DT%.log"
