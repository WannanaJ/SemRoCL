@echo off
REM ============================================
REM SemRoCL 多数据集消融实验批处理脚本
REM Multi-Dataset Ablation Experiments Batch Script
REM ============================================

echo.
echo ========================================
echo SemRoCL Multi-Dataset Ablation Study
echo ========================================
echo.

REM 设置数据集列表
set DATASETS=Endo-LowLight ExDark Learning-to-see-in-the-dark LOLBlur MEF SICE

REM 进入实验目录
cd /d "%~dp0..\src\experiments"

echo 将在以下6个数据集上运行实验:
echo 1. Endo-LowLight (内窥镜低光照)
echo 2. ExDark (极低光照)
echo 3. Learning-to-see-in-the-dark (RAW图像)
echo 4. LOLBlur (低光照+模糊)
echo 5. MEF (多曝光融合)
echo 6. SICE (多曝光增强)
echo.
echo 预计总时间: 6-12小时 (取决于GPU性能)
echo.

pause

REM 遍历每个数据集
for %%D in (%DATASETS%) do (
    echo.
    echo ========================================
    echo 正在处理数据集: %%D
    echo ========================================
    echo.

    REM 创建临时配置文件
    python create_dataset_config.py --dataset %%D --base-config ..\..\configs\experiment_multi_dataset.yaml

    if errorlevel 1 (
        echo 错误: 无法创建配置文件 for %%D
        goto :error
    )

    REM 步骤1: 消融实验
    echo [1/4] 运行消融实验...
    python run_ablation.py --config ..\..\configs\temp_%%D.yaml
    if errorlevel 1 (
        echo 错误: 消融实验失败 for %%D
        goto :error
    )

    REM 步骤2: 计算指标
    echo [2/4] 计算评估指标...
    python eval_metrics.py --config ..\..\configs\temp_%%D.yaml
    if errorlevel 1 (
        echo 错误: 指标计算失败 for %%D
        goto :error
    )

    REM 步骤3: 生成可视化
    echo [3/4] 生成可视化图表...
    python visualize_results.py --config ..\..\configs\temp_%%D.yaml
    if errorlevel 1 (
        echo 错误: 可视化失败 for %%D
        goto :error
    )

    REM 步骤4: 生成LaTeX表格
    echo [4/4] 生成LaTeX表格...
    python summarize_results.py --config ..\..\configs\temp_%%D.yaml
    if errorlevel 1 (
        echo 错误: 结果汇总失败 for %%D
        goto :error
    )

    REM 清理临时配置
    del ..\..\configs\temp_%%D.yaml

    echo.
    echo 数据集 %%D 完成!
    echo.
)

REM 步骤5: 跨数据集汇总
echo.
echo ========================================
echo [5/5] 生成跨数据集汇总报告...
echo ========================================
python summarize_cross_dataset.py --base-dir ..\..\outputs\ablation_multi_dataset

if errorlevel 1 (
    echo 警告: 跨数据集汇总失败
)

echo.
echo ========================================
echo 所有实验完成!
echo ========================================
echo.
echo 结果保存在: outputs\ablation_multi_dataset\
echo.
echo 各数据集结果:
for %%D in (%DATASETS%) do (
    echo   - %%D: outputs\ablation_multi_dataset\%%D\
)
echo.
echo 跨数据集汇总: outputs\ablation_multi_dataset\cross_dataset_summary\
echo.

pause
goto :end

:error
echo.
echo ========================================
echo 实验过程中出现错误!
echo ========================================
echo.
pause
exit /b 1

:end
