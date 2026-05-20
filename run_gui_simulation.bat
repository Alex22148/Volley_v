@echo off
cd /d "%~dp0"
set VOLLEYHUB_CONSOLE=1
set VOLLEYHUB_SIMULATE=1
".venv\Scripts\python.exe" -u main.py
echo.
echo VolleyHub exited with code %ERRORLEVEL%.
pause
