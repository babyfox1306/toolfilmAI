@echo off
REM ToolfilmAI Startup Script
SETLOCAL EnableExtensions

echo ===== TOOLFILM AI =====
echo Starting application...

REM Force GPU settings with compatible CUDA version
SET FORCE_CUDA=1
SET CUDA_VISIBLE_DEVICES=0
SET PYTHONPATH=%cd%
SET PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
SET ULTRALYTICS_DEVICE=0

REM Check CUDA installation
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo Found CUDA installation
    for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        echo Setting CUDA_HOME to: %%G
        SET CUDA_HOME=%%G
    )
)

REM Check Python and PyTorch with GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" || (
    echo Failed to check PyTorch CUDA availability
)

REM Làm sạch bộ nhớ GPU
taskkill /F /IM "nvidia-smi.exe" /T > nul 2>&1

REM Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python first.
    pause
    exit /b
)

REM Check if virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment. Please check your Python installation.
        pause
        exit /b
    )
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if requirements are installed
if not exist .venv\installed.flag (
    echo Installing requirements...
    pip install -r requirements.txt
    echo. > .venv\installed.flag
)

REM Debug GPU
echo Running GPU debug...
python debug_gpu.py

REM Run the main application
python main.py

REM Deactivate virtual environment
call .venv\Scripts\deactivate.bat

echo Application closed.
pause
ENDLOCAL