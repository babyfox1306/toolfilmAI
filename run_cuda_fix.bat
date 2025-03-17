@echo off
REM CUDA and PyTorch Installation Fix for ToolfilmAI
SETLOCAL EnableExtensions

echo ===== TOOLFILM AI - CUDA & PYTORCH INSTALLATION FIX =====
echo This script will diagnose and fix CUDA and PyTorch installation issues.

REM Set CUDA environment
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo Found CUDA installation
    for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        echo Setting CUDA_HOME to: %%G
        SET CUDA_HOME=%%G
    )
)

REM Set PYTHONPATH
SET PYTHONPATH=%cd%

REM Activate virtual environment if exists
if exist .venv (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the CUDA fix script
python fix_cuda_install.py

REM Deactivate virtual environment
if exist .venv (
    call .venv\Scripts\deactivate.bat
)

echo.
echo ===== FINISHED =====
echo If PyTorch was reinstalled, please restart your Python environment before running the application.
pause
ENDLOCAL
