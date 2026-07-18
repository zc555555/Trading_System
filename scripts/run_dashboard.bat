@echo off
REM ===========================================================================
REM Launch the Streamlit live trading dashboard at http://localhost:8501
REM ===========================================================================
chcp 65001 >nul
cd /d "%~dp0\.."

echo ================================================================================
echo Starting Real-Time Trading Dashboard
echo ================================================================================
echo   URL:    http://localhost:8501
echo   Stop:   Ctrl+C
echo ================================================================================
echo.

streamlit run dashboard.py
pause
