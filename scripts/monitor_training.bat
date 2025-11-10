@echo off
REM Real-time monitoring of Stage 2 training

echo ============================================================
echo  SemRoCL Training Monitor
echo ============================================================
echo.

set LOG_DIR=outputs\semantic_enhancement_stage2_ENHANCED\logs

if not exist "%LOG_DIR%" (
    echo [ERROR] Log directory not found: %LOG_DIR%
    echo Please check the training output path.
    pause
    exit /b 1
)

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo [INFO] Monitoring training logs in: %LOG_DIR%
echo [INFO] Press Ctrl+C to stop monitoring
echo.

python tools\monitor_training.py --log_dir "%LOG_DIR%" --refresh 10

pause
