@echo off
REM Resume Stage 2 training with optimized configuration
REM This will use the new optimized config while loading from existing checkpoint

echo ============================================================
echo  SemRoCL Stage 2 Training - OPTIMIZED Resume
echo ============================================================
echo.

REM Check if checkpoint exists
if not exist "outputs\semantic_enhancement_stage2_ENHANCED\checkpoints\latest.pth" (
    echo [ERROR] No checkpoint found!
    echo Please run initial training first or check the checkpoint path.
    pause
    exit /b 1
)

echo [INFO] Found existing checkpoint
echo [INFO] Starting with OPTIMIZED configuration...
echo.
echo Key optimizations:
echo   - num_workers: 0 --^> 4 (30-40%% speed boost)
echo   - persistent_workers enabled
echo   - Enhanced color loss weights
echo   - Better prefetching
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run training with optimized config
echo [INFO] Launching training...
python src\train_stage2_enhanced.py ^
    --config configs\train_stage2_enhanced_optimized.yaml ^
    --resume outputs\semantic_enhancement_stage2_ENHANCED\checkpoints\latest.pth

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Training completed successfully!
echo ============================================================
pause
