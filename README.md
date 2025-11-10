# SemRoCL: Semantic-guided Robust Contrastive Learning Framework

[![Tests](https://github.com/your-username/SemRoCL/workflows/Tests/badge.svg)](https://github.com/your-username/SemRoCL/actions)
[![Code Quality](https://github.com/your-username/SemRoCL/workflows/Code%20Quality/badge.svg)](https://github.com/your-username/SemRoCL/actions)
[![Documentation](https://github.com/your-username/SemRoCL/workflows/Documentation/badge.svg)](https://semrocl.readthedocs.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A two-stage unsupervised framework for low-light image enhancement combining contrastive learning, semantic guidance, and multi-objective optimization.

---

## 🎯 Project Goals

- **Unsupervised Learning**: Eliminate dependency on paired training data
- **Semantic Guidance**: Leverage scene understanding for targeted enhancement
- **Robustness**: Handle extreme illumination and noise conditions
- **Interpretability**: Provide explainable semantic heatmaps
- **Lightweight**: Enable deployment on edge devices by 2026

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🚀 Installation

### 1. Create Conda Environment

```bash
cd D:\projects\SemRoCL
conda env create -f environment.yml
conda activate semrocl
```

### 2. Alternative: pip Installation

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Required Datasets

1. **LOL-v1** - Low-Light dataset with paired images
   - Download from: https://daooshee.github.io/BMVC2018website/
   - Place in: `data/LOL-v1/`

2. **LOL-v2** - Enhanced low-light dataset
   - Download from: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
   - Place in: `data/LOL-v2/`

3. **LIME** - Real-world low-light images
   - Download from: https://sites.google.com/view/chen-wei-homepage/datasets
   - Place in: `data/LIME/`

### Dataset Structure

```
data/
├── LIME/
├── LOL-v1/
│   ├── eval15/
│   │   ├── high/
│   │   └── low/
│   └── our485/
│       ├── high/
│       └── low/
└── LOL-v2/
    ├── Real_captured/
    │   ├── Test/
    │   └── Train/
    └── Synthetic/
        ├── Test/
        └── Train/
```

## 🏋️ Training

### Stage 1: MoCo v3 Contrastive Pretraining

Train the encoder unsupervisedly on low-light images:

```bash
cd src
python train_stage1_moco.py --config ../configs/train_stage1.yaml
```

**Key Features:**
- Unsupervised contrastive learning (无监督对比学习)
- Dynamic queue for negative samples (动态负样本队列)
- Illumination and noise augmentations (光照和噪声增强)
- Outputs: `outputs/moco_pretrain_stage1/checkpoints/moco_pretrain_final.pth`

**Lightweight Version** (推荐用于快速迭代):
```bash
python train_stage1_moco.py --config ../configs/train_stage1_light.yaml
```
- ResNet-18 backbone (~45% faster training)
- Smaller queue size and image resolution

### Stage 2: Semantic-Guided Enhancement

Fine-tune the enhancer with semantic guidance and adaptive curriculum learning:

```bash
python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced.yaml
```

**Key Features:**
- Multi-scale enhancement (多尺度增强)
- SegFormer-B0 semantic guidance with confidence masking (语义引导与置信度掩码)
- Adaptive curriculum learning (自适应课程学习)
- Multi-objective optimization (多目标优化):
  - L_color: Exposure control + color constancy + total variation
  - L_semantic: Semantic consistency weighted by uncertainty
  - L_perceptual: VGG-based perceptual loss
  - L_frequency: DCT-based frequency-domain consistency
  - L_adv: PatchGAN adversarial loss (after epoch 90)
  - L_noise: Denoising loss for artifact reduction (新增降噪损失)

**Optimized Training** (优化配置):
```bash
python train_stage2_enhanced.py --config ../configs/train_stage2_enhanced_optimized.yaml
# OR use the batch script:
scripts\resume_training_optimized.bat
```
- 30-40% faster data loading (num_workers: 4, persistent_workers)
- Enhanced color loss weights for better color fidelity
- Comprehensive metrics logging (PSNR, SSIM, DeltaE, NIQE, BRISQUE)

**Curriculum Training** (课程学习):
```bash
python train_stage2_curriculum.py --config ../configs/train_stage2_curriculum.yaml
```
- Progressive phase-based training
- Smooth loss weight transitions
- Automatic performance monitoring

## 📈 Evaluation

Evaluate on test datasets with multiple metrics:

```bash
python evaluate.py \
    --config ../configs/train_stage2_enhanced.yaml \
    --checkpoint ../outputs/semantic_enhancement_stage2_ENHANCED/checkpoints/best_model.pth
```

**Metrics Computed:**
- PSNR (Peak Signal-to-Noise Ratio) - 图像质量
- SSIM (Structural Similarity Index) - 结构相似性
- LPIPS (Learned Perceptual Image Patch Similarity) - 感知相似性
- DeltaE (Color Difference) - 色彩差异
- NIQE (Natural Image Quality Evaluator) - 自然图像质量
- BRISQUE (Blind/Referenceless Image Spatial Quality) - 无参考质量评估

### 🎨 Image Quality Optimization

本项目包含图像后处理优化，减少输出图像的伪影和噪点：

**优化技术** (Optimization Techniques):
1. **双边滤波** (Bilateral Filtering) - 保留边缘的同时去噪
2. **非局部均值降噪** (Non-local Means Denoising) - 智能噪点移除
3. **自适应锐化** (Adaptive Sharpening) - 补偿降噪模糊
4. **高质量保存** (High-Quality Saving):
   - PNG: 最小压缩 (compress_level=1)
   - JPEG: 高质量无色度子采样 (quality=95, subsampling=0)

**使用方法** (Usage):
```python
from utils import save_image

# 自动应用所有优化
save_image(tensor, 'output.png', quality=95, apply_post_processing=True)

# 禁用后处理（更快但质量较低）
save_image(tensor, 'output.png', apply_post_processing=False)
```

## 📁 Project Structure

```
SemRoCL/
├── data/                          # 数据集 / Datasets
│   ├── LOL-v1/                    # LOL dataset v1
│   ├── LOL-v2/                    # LOL dataset v2
│   └── LIME/                      # LIME dataset
├── src/                           # 源代码 / Source code
│   ├── model/                     # 模型架构 / Model architectures
│   │   ├── encoder_moco.py        # MoCo v3 encoder
│   │   ├── semantic_head.py       # SegFormer semantic guidance
│   │   ├── models.py              # Base generator & discriminator
│   │   ├── models_enhanced.py     # Enhanced multi-scale generator
│   │   ├── enhancer.py            # Curve-based enhancer
│   │   └── discriminator.py       # PatchGAN discriminator
│   ├── data_loader.py             # 数据加载器 / Dataset loader
│   ├── loss_functions.py          # 多目标损失函数 / Multi-objective losses
│   ├── adaptive_curriculum.py     # 自适应课程学习 / Adaptive curriculum
│   ├── metrics_utils.py           # 评估指标工具 / Metrics utilities
│   ├── train_stage1_moco.py       # Stage 1 训练 / Stage 1 training
│   ├── train_stage2_enhanced.py   # Stage 2 增强训练 / Stage 2 enhanced training
│   ├── train_stage2_curriculum.py # Stage 2 课程训练 / Stage 2 curriculum training
│   ├── evaluate.py                # 评估脚本 / Evaluation script
│   └── utils.py                   # 工具函数 / Helper functions (with image denoising)
├── configs/                       # 配置文件 / Configuration files
│   ├── train_stage1.yaml          # Stage 1 full config
│   ├── train_stage1_light.yaml    # Stage 1 lightweight config
│   ├── train_stage2_enhanced.yaml # Stage 2 enhanced config
│   ├── train_stage2_enhanced_optimized.yaml  # Stage 2 optimized config (recommended)
│   ├── train_stage2_enhanced_light.yaml      # Stage 2 lightweight config
│   └── train_stage2_curriculum.yaml          # Stage 2 curriculum config
├── scripts/                       # 批处理脚本 / Batch scripts
│   ├── resume_training_optimized.bat  # Resume optimized training
│   ├── monitor_training.bat           # Monitor training progress
│   ├── plot_metrics.bat               # Plot training curves
│   └── *.py                           # Analysis utilities
├── tools/                         # 工具脚本 / Utility scripts
│   ├── diagnose_training.py       # Training diagnostics
│   ├── monitor_training.py        # Real-time training monitor
│   ├── plot_training_curves.py    # Plot training metrics
│   └── compare_configs.py         # Compare config files
├── docs/                          # 文档 / Documentation
│   ├── MODEL_OVERVIEW.md          # Model architecture overview
│   ├── LOSSES_AND_WEIGHTS.md      # Loss function details
│   └── OPTIMIZATION_GUIDE.md      # Performance optimization guide
├── outputs/                       # 输出 / Outputs (checkpoints, logs, images)
├── experiments/                   # 实验记录 / Experiment logs
├── environment.yml                # Conda environment
├── requirements.txt               # pip requirements (优化版 / optimized)
└── README.md                      # This file
```

## 🔬 Methodology

### Two-Stage Pipeline

**Stage 1: Robust Feature Pretraining**
```
[Low-Light Input] → [MoCo v3 + Augmentation] → [Universal Encoder]
```

**Stage 2: Semantic-Guided Enhancement**
```
[Frozen Encoder] → [SegFormer-B0 + Uncertainty] → [Curve Enhancer] → [Enhanced Output]
```

### Multi-Objective Loss Function

```
L_total = λ₁·L_color + λ₂·L_semantic + λ₃·L_frequency + λ₄·L_adv + λ₅·L_task
```

- **L_color**: Exposure control + color constancy + total variation
- **L_semantic**: Semantic consistency weighted by uncertainty
- **L_frequency**: DCT-based frequency-domain consistency
- **L_adv**: PatchGAN adversarial loss
- **L_task**: Feature similarity using pretrained backbone

## 📊 Expected Results

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ |
|---------|--------|--------|---------|--------|
| LOL-v1  | 24-26  | 0.85+  | 0.15-   | 3.5-   |
| LOL-v2  | 23-25  | 0.82+  | 0.18-   | 3.8-   |
| LIME    | N/A    | N/A    | 0.20-   | 4.0-   |

## 🚀 Deployment

### Lightweight Model Export

```bash
python src/export_onnx.py --checkpoint <path> --output model.onnx
```

### TensorRT Optimization

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{semrocl2026,
  title={SemRoCL: Semantic-guided Robust Contrastive Learning for Low-Light Image Enhancement},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com

## 🙏 Acknowledgments

- MoCo v3: Momentum Contrast for Unsupervised Learning
- SegFormer: Simple and Efficient Design for Semantic Segmentation
- Zero-DCE: Zero-Reference Deep Curve Estimation
- LOL Dataset: Low-Light Image Enhancement Dataset

---

**Status**: Active Development | **Target**: Deployment Ready by 2026
