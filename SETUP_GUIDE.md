# SemRoCL Setup Guide

## Quick Start

### 1. Clone/Copy Project to D:\projects\SemRoCL

Copy the entire SemRoCL folder to your D drive at:
```
D:\projects\SemRoCL\
```

### 2. Environment Setup

**Option A: Conda (Recommended)**
```bash
cd D:\projects\SemRoCL
conda env create -f environment.yml
conda activate semrocl
```

**Option B: pip + venv**
```bash
cd D:\projects\SemRoCL
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Download Datasets

Download and organize datasets as follows:

```
D:\projects\SemRoCL\data\
├── LIME\                  # Download from: https://sites.google.com/view/chen-wei-homepage/datasets
├── LOL-v1\               # Download from: https://daooshee.github.io/BMVC2018website/
│   ├── eval15\
│   │   ├── high\
│   │   └── low\
│   └── our485\
│       ├── high\
│       └── low\
└── LOL-v2\               # Download from: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
    ├── Real_captured\
    │   ├── Test\
    │   └── Train\
    └── Synthetic\
        ├── Test\
        └── Train\
```

### 4. Verify Installation

```bash
cd src
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Training Pipeline

### Stage 1: Contrastive Pretraining (Unsupervised)

```bash
cd D:\projects\SemRoCL\src
python train_stage1_moco.py --config ../configs/train_stage1.yaml
```

**Expected Output:**
- Checkpoints saved to: `outputs/moco_pretrain_stage1/checkpoints/`
- Training logs: `outputs/moco_pretrain_stage1/logs/`
- Training time: ~20-30 hours on RTX 3090

### Stage 2: Enhancement Training (Supervised)

```bash
python train_stage2_enhance.py --config ../configs/train_stage2.yaml
```

**Expected Output:**
- Enhanced model checkpoints: `outputs/semantic_enhancement_stage2/checkpoints/`
- Sample visualizations: `outputs/semantic_enhancement_stage2/results/`
- Training time: ~15-20 hours on RTX 3090

### Evaluation

```bash
python evaluate.py \
    --config ../configs/train_stage2.yaml \
    --checkpoint ../outputs/semantic_enhancement_stage2/checkpoints/stage2_epoch_100.pth
```

## Configuration

### Modify Training Parameters

Edit configuration files in `configs/`:

**train_stage1.yaml** - MoCo pretraining:
```yaml
training:
  epochs: 200          # Reduce for faster training
  batch_size: 32       # Adjust based on GPU memory
  lr: 0.03             # Learning rate
```

**train_stage2.yaml** - Enhancement training:
```yaml
training:
  epochs: 100
  batch_size: 16
  lr: 0.0001

loss:
  w_color: 1.0         # Color consistency weight
  w_semantic: 1.0      # Semantic guidance weight
  w_adv: 0.1           # Adversarial weight
  w_freq: 0.5          # Frequency domain weight
  w_task: 0.5          # Task-specific weight
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Reduce batch_size in config files
- Use smaller image size (e.g., 384 instead of 512)
- Enable gradient checkpointing

### Issue: Dataset Not Found

**Solution:**
- Verify dataset paths in config files
- Check folder structure matches expected layout
- Ensure images are in standard formats (PNG, JPG)

### Issue: Slow Training

**Solution:**
- Increase num_workers in DataLoader (if CPU allows)
- Enable mixed precision training (add to config)
- Use smaller model variants (resnet18 instead of resnet50)

## Hardware Requirements

### Minimum
- GPU: NVIDIA RTX 2060 (6GB VRAM)
- RAM: 16GB
- Storage: 50GB free space

### Recommended
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

### For Inference Only
- GPU: NVIDIA GTX 1660 (6GB VRAM)
- RAM: 8GB
- Storage: 10GB

## Next Steps

1. ✅ Environment setup complete
2. ✅ Datasets downloaded and organized
3. ⬜ Run Stage 1 training (MoCo pretraining)
4. ⬜ Run Stage 2 training (Enhancement)
5. ⬜ Evaluate on test datasets
6. ⬜ Deploy lightweight models

## Support

- GitHub Issues: https://github.com/semrocl/SemRoCL/issues
- Documentation: `docs/README_Research.md`
- Research Plan: `docs/Low-Light_Enhancement_Research_Plan_SemRoCL_2026.docx`

## License

MIT License - See LICENSE file for details
