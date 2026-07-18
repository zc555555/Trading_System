@echo off
REM ===========================================================================
REM Interactive control menu (recommended entry point for manual operation).
REM ===========================================================================
chcp 65001 >nul
cd /d "%~dp0\.."
python run_menu.py
pause
