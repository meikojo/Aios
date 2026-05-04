@echo off
title AIOS Core - Boot Menu
color 0A
cd /d "%~dp0"

echo [1] Checking Virtual Environment...
if not exist "venv\Scripts\python.exe" (
    echo Creating clean environment...
    python -m venv venv
)

echo [2] Forcing installation of required AI libraries...
"%~dp0venv\Scripts\python.exe" -m pip install safetensors torch numpy psutil PyQt6

echo.
echo [3] Launching AIOS Core Engine...
"%~dp0venv\Scripts\python.exe" app.py

pause