@echo off
rem Unattended weekly retrain entry (Task Scheduler: Sunday 20:00 UK).
rem Failures land in trading_logs\FAILURES.log and raise a desktop notice.
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
echo ================ retrain started %date% %time% ================ >> "trading_logs\retrain_%DT%.log"
"research\venv\Scripts\python.exe" run_retrain_models.py >> "trading_logs\retrain_%DT%.log" 2>&1
set RC=%errorlevel%
echo ================ retrain finished %date% %time% exit=%RC% ================ >> "trading_logs\retrain_%DT%.log"
rem Weekly slippage / execution-A/B reconciliation (non-fatal)
"research\venv\Scripts\python.exe" research\evaluation\experiments\slippage_calibration.py >> "trading_logs\retrain_%DT%.log" 2>&1
if not "%RC%"=="0" (
    echo %date% %time% WEEKLY RETRAIN FAILED exit=%RC% see retrain_%DT%.log >> "trading_logs\FAILURES.log"
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0notify_failure.ps1" "StockPredict 周度重训失败" "exit=%RC%, 详见 trading_logs\retrain_%DT%.log"
)
