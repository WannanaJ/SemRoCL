@echo off
REM Compare original and optimized configurations

echo ============================================================
echo  Configuration Comparison Tool
echo ============================================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Comparing configurations...
echo.

python tools\compare_configs.py ^
    --config1 configs\train_stage2_enhanced.yaml ^
    --config2 configs\train_stage2_enhanced_optimized.yaml

echo.
pause
