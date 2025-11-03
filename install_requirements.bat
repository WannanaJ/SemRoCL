@echo off
echo ============================================================
echo 🚀 Installing Python dependencies for SemRoCL environment...
echo ============================================================

:: 激活 Conda 环境（请确认环境名 semrocl 正确）
call conda activate semrocl

:: 基础科学计算库
pip install numpy scipy pandas matplotlib seaborn tqdm scikit-learn

:: 深度学习框架
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 图像增强与读取库
pip install opencv-python pillow albumentations

:: 低光照研究常用工具
pip install scikit-image imageio tensorboard

:: 数据加载与可视化
pip install plotly seaborn

:: 文件操作与兼容性支持
pip install pyyaml rich colorama

:: 检查安装状态
echo ============================================================
echo ✅ Dependency installation completed!
echo You can now run:
echo     python tools/visualize_moco_features.py --help
echo ============================================================

pause
