@echo off
cd /d "%~dp0"
set "PY=python"
if exist "server\venv\Scripts\python.exe" set "PY=server\venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if exist "..\so-vits-svc\venv\Scripts\python.exe" set "PY=..\so-vits-svc\venv\Scripts\python.exe"
echo Starting So-VITS-SVC server...
echo URL: http://localhost:5000
echo Press Ctrl+C to stop.
echo.
"%PY%" server\app.py
echo.
echo Server stopped.
pause
