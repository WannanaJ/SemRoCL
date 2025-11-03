# SemRoCL: Semantic-guided Robust Contrastive Learning Framework

A two-stage unsupervised framework for low-light image enhancement combining contrastive learning, semantic guidance, and multi-objective optimization.

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
- Unsupervised contrastive learning
- Dynamic queue for negative samples
- Illumination and noise augmentations
- Outputs: `outputs/checkpoints/moco_pretrain_final.pth`

### Stage 2: Semantic-Guided Enhancement

Fine-tune the enhancer with semantic guidance:

```bash
python train_stage2_enhance.py --config ../configs/train_stage2.yaml
```

**Key Features:**
- Frozen pretrained encoder
- SegFormer-B0 semantic guidance
- Multi-objective optimization (L_color + L_semantic + L_frequency + L_adv + L_task)
- Uncertainty-weighted losses

## 📈 Evaluation

Evaluate on test datasets with multiple metrics:

```bash
python evaluate.py \
    --config ../configs/train_stage2.yaml \
    --checkpoint ../outputs/semantic_enhancement_stage2/checkpoints/stage2_epoch_100.pth
```

**Metrics Computed:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- NIQE (Natural Image Quality Evaluator)

## 📁 Project Structure

```
SemRoCL/
├── data/                      # Datasets
├── src/                       # Source code
│   ├── model/                 # Model architectures
│   │   ├── encoder_moco.py    # MoCo v3 encoder
│   │   ├── semantic_head.py   # SegFormer-B0 + uncertainty
│   │   ├── enhancer.py        # Curve-based enhancer
│   │   └── discriminator.py   # PatchGAN discriminator
│   ├── data_loader.py         # Dataset loader
│   ├── loss_functions.py      # Multi-objective losses
│   ├── train_stage1_moco.py   # Stage 1 training
│   ├── train_stage2_enhance.py # Stage 2 training
│   ├── evaluate.py            # Evaluation script
│   └── utils.py               # Helper functions
├── configs/                   # Configuration files
├── outputs/                   # Checkpoints, logs, results
├── docs/                      # Documentation
├── environment.yml            # Conda environment
├── requirements.txt           # pip requirements
└── README.md                  # This file
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
