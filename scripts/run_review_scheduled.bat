@echo off
rem Segment-rotation review (RULEBOOK v3, REVIEW_SCHEDULE). Task Scheduler
rem fires it on the review date at 20:00 (S4U); the date is passed as %1 so
rem the same script serves every scheduled review. Steps, all at both
rem horizons: refresh the surv PIT panel, re-run the incumbent baseline,
rem re-run every track-B full-stage candidate and re-adjudicate under the
rem rotated segments. Several hours; output in research\evaluation\results\
rem review_<date>_h<H>.md. Nothing is adopted automatically.
cd /d "%~dp0.."
set RD=%1
if "%RD%"=="" for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set RD=%%i
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set DT=%%i
if not exist trading_logs mkdir trading_logs
set LOG=trading_logs\review_%DT%.log
set PYTHONIOENCODING=utf-8
echo ================ review %RD% started %date% %time% ================ >> "%LOG%"
"research\venv\Scripts\python.exe" research\data\fetch_sharadar_prices.py --refresh >> "%LOG%" 2>&1
"research\venv\Scripts\python.exe" research\evaluation\experiments\survivorship_universe.py --mode surv >> "%LOG%" 2>&1
for %%H in (20 5) do (
    "research\venv\Scripts\python.exe" research\mining\harness.py --horizon %%H baseline >> "%LOG%" 2>&1
    "research\venv\Scripts\python.exe" research\mining\harness.py --horizon %%H review --date %RD% --rerun >> "%LOG%" 2>&1
)
set RC=%errorlevel%
echo ================ review finished %date% %time% exit=%RC% ================ >> "%LOG%"
if not "%RC%"=="0" (
    echo %date% %time% SEGMENT REVIEW FAILED exit=%RC% see %LOG% >> "trading_logs\FAILURES.log"
)
