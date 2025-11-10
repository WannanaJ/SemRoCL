@echo off
REM Generate comprehensive metrics plots

echo ============================================================
echo  SemRoCL Metrics Visualization
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

echo [INFO] Generating plots from: %LOG_DIR%
echo.

python tools\monitor_training.py --log_dir "%LOG_DIR%" --plot --save_plot "%LOG_DIR%\training_metrics.png"

if errorlevel 1 (
    echo.
    echo [ERROR] Plot generation failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Plots saved to: %LOG_DIR%\training_metrics.png
echo Opening plot...
start "" "%LOG_DIR%\training_metrics.png"

pause
