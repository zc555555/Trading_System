@echo off
rem Monthly refresh of the mining DSL's external sources (Task Scheduler:
rem 25th of each month 19:00, S4U). SEC insider datasets (new quarters only),
rem FINRA short interest, GDELT current year, then both screen caches.
rem Not part of production; failures land in trading_logs\FAILURES.log.
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
set LOG=trading_logs\refresh_sources_%DT%.log
echo ================ refresh started %date% %time% ================ >> "%LOG%"
set PYTHONIOENCODING=utf-8
"research\venv\Scripts\python.exe" research\data\refresh_mining_sources.py >> "%LOG%" 2>&1
set RC=%errorlevel%
echo ================ refresh finished %date% %time% exit=%RC% ================ >> "%LOG%"
if not "%RC%"=="0" (
    echo %date% %time% MINING SOURCE REFRESH FAILED exit=%RC% see %LOG% >> "trading_logs\FAILURES.log"
)
