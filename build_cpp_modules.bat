@echo off
echo Building C++ extensions for ToolfilmAI...

REM Check for Python and set environment variables
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXECUTABLE=%%i
for /f "delims=" %%i in ('python -c "import sys; print(sys.prefix)"') do set PYTHON_HOME=%%i
for /f "delims=" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
echo Python executable: %PYTHON_EXECUTABLE%
echo Python home: %PYTHON_HOME%
echo Python include: %PYTHON_INCLUDE%

REM Find CUDA
if exist "%CUDA_HOME%" (
    echo CUDA_HOME is set to: %CUDA_HOME%
) else (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
        for /d %%G in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
            echo Found CUDA: %%G
            set CUDA_HOME=%%G
        )
    )
)

REM Check for CMake
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo CMake not found. Please install CMake and add it to PATH.
    exit /b 1
)

REM Create build directory if it doesn't exist
if not exist build mkdir build
cd build

REM Configure CMake
echo Configuring CMake...
cmake -G "Visual Studio 16 2019" -A x64 -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" -DPython3_ROOT_DIR="%PYTHON_HOME%" ..\cpp
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed.
    cd ..
    exit /b 1
)

REM Build the extensions
echo Building extensions...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    cd ..
    exit /b 1
)

REM Copy the built extensions to the main directory
echo Copying built extensions...
copy /Y Release\optical_flow_cpp.pyd ..\
copy /Y Release\video_extract_cpp.pyd ..\
copy /Y Release\video_concat_cpp.pyd ..\

cd ..
echo Build completed successfully.
