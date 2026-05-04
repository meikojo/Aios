@echo off
title AIOS Core - Ultimate Boot
color 0B

echo ===================================================
echo   AIOS Core Engine - Clean Boot Sequence
echo ===================================================
echo.

echo [1] Sweeping old compiled files and caches...
FOR /d /r . %%d IN (__pycache__) DO @IF EXIST "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
del /s /q *.pyo >nul 2>&1

echo [2] Forcing installation of required libraries...
python -m pip install safetensors torch numpy psutil PyQt6 --force-reinstall

echo.
echo [3] Initializing Core Engine...
echo ===================================================
echo.

:: تشغيل البرنامج بالنسخة الأساسية للويندوز (3.14)
python app.py

echo.
echo ===================================================
echo Engine gracefully shut down.
pause