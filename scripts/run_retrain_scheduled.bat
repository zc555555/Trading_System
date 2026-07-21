@echo off
rem Unattended weekly retrain entry (Task Scheduler: Sunday 20:00 UK).
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
echo ================ retrain started %date% %time% ================ >> "trading_logs\retrain_%DT%.log"
"research\venv\Scripts\python.exe" run_retrain_models.py >> "trading_logs\retrain_%DT%.log" 2>&1
echo ================ retrain finished %date% %time% exit=%errorlevel% ================ >> "trading_logs\retrain_%DT%.log"
