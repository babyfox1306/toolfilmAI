@echo off
REM YOLOv8 GPU Fix for ToolfilmAI
SETLOCAL EnableExtensions

echo ===== TOOLFILM AI - YOLOv8 GPU FIX =====
echo Applying YOLOv8 GPU fixes...

REM Force GPU settings with compatible CUDA version
SET FORCE_CUDA=1
SET CUDA_VISIBLE_DEVICES=0
SET PYTHONPATH=%cd%
SET PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
SET ULTRALYTICS_DEVICE=cuda:0

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

REM Activate virtual environment
if exist .venv (
    call .venv\Scripts\activate.bat
)

REM Run the patched application
python fix_yolo8_gpu.py

REM Deactivate virtual environment
if exist .venv (
    call .venv\Scripts\deactivate.bat
)

echo Application closed.
pause
ENDLOCAL
