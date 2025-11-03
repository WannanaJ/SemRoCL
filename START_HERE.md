# 🚀 START HERE - SemRoCL Project Quick Start

## Welcome to SemRoCL!

This is a complete deep-learning research project for **Low-Light Image Enhancement** using semantic-guided contrastive learning.

## 📦 What's Included?

✅ **Complete Source Code** - All model components, training scripts, and utilities  
✅ **Configuration Files** - Ready-to-use YAML configs for all experiments  
✅ **Documentation** - Research plan, setup guides, and API documentation  
✅ **Dataset Structure** - Organized folders for LOL-v1, LOL-v2, and LIME  
✅ **Pre-configured Environment** - Conda and pip environment specifications  

## 🎯 Quick Start (5 Steps)

### Step 1: Copy to Your Machine
```
Copy the entire SemRoCL folder to:
D:\projects\SemRoCL\
```

### Step 2: Setup Environment
```bash
cd D:\projects\SemRoCL
conda env create -f environment.yml
conda activate semrocl
```

### Step 3: Download Datasets
Download and place in `data/` folder:
- **LOL-v1**: https://daooshee.github.io/BMVC2018website/
- **LOL-v2**: https://github.com/flyywh/CVPR-2020-Semi-Low-Light  
- **LIME**: https://sites.google.com/view/chen-wei-homepage/datasets

### Step 4: Train Stage 1 (MoCo Pretraining)
```bash
cd src
python train_stage1_moco.py --config ../configs/train_stage1.yaml
```

### Step 5: Train Stage 2 (Enhancement)
```bash
python train_stage2_enhance.py --config ../configs/train_stage2.yaml
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **README.md** | Main project documentation with full details |
| **SETUP_GUIDE.md** | Detailed setup and troubleshooting guide |
| **PROJECT_STRUCTURE.txt** | Complete project structure visualization |
| **docs/README_Research.md** | 18-month research plan overview |
| **docs/Research_Plan.docx** | Formal research plan document |

## 🏗️ Project Architecture

```
Two-Stage Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Unsupervised Contrastive Pretraining              │
│ [Low-Light] → [MoCo v3] → [Universal Encoder]              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Semantic-Guided Enhancement                        │
│ [Frozen Encoder] → [SegFormer-B0] → [Curve Enhancer]       │
│                  ↓                                           │
│              [Enhanced Image]                                │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Key Innovations

1. **Unsupervised Learning** - No paired data needed for pretraining
2. **Semantic Guidance** - Scene-aware enhancement
3. **Uncertainty Weighting** - Confidence-based dynamic optimization
4. **Multi-Objective Loss** - Balanced enhancement quality
5. **Lightweight Design** - Edge-deployable architecture

## 📊 Expected Performance

| Metric | Target |
|--------|--------|
| PSNR | 24-26 dB |
| SSIM | 0.85+ |
| LPIPS | < 0.15 |
| Inference | < 50ms (GPU) |

## 🔧 Customization

Edit these files to customize:
- `configs/train_stage1.yaml` - MoCo pretraining parameters
- `configs/train_stage2.yaml` - Enhancement training parameters
- `configs/model_config.yaml` - Model architecture settings

## 📈 Training Time Estimates

| Stage | GPU | Time |
|-------|-----|------|
| Stage 1 | RTX 3090 | ~24 hours |
| Stage 2 | RTX 3090 | ~16 hours |
| Total | RTX 3090 | ~40 hours |

## ❓ Need Help?

1. **Setup Issues?** → Read `SETUP_GUIDE.md`
2. **Understanding the Code?** → Check `PROJECT_STRUCTURE.txt`
3. **Research Context?** → Open `docs/Research_Plan.docx`
4. **Technical Details?** → See `README.md`

## 🎓 Research Plan

This project follows an 18-month research roadmap:
- **Months 1-6**: Framework construction and validation
- **Months 7-12**: Robustness and lightweight deployment
- **Months 13-18**: Multimodal expansion and publication

See `docs/Low-Light_Enhancement_Research_Plan_SemRoCL_2026.docx` for complete details.

## 🚀 Next Steps

1. ✅ Read this file (you're here!)
2. ⬜ Setup environment (`SETUP_GUIDE.md`)
3. ⬜ Download datasets (`data/README.md`)
4. ⬜ Run Stage 1 training
5. ⬜ Run Stage 2 training
6. ⬜ Evaluate results
7. ⬜ Deploy models

## 📝 Citation

If you use this code, please cite:
```bibtex
@article{semrocl2026,
  title={SemRoCL: Semantic-guided Robust Contrastive Learning 
         for Low-Light Image Enhancement},
  year={2026}
}
```

## 📧 Support

Questions? Issues? Ideas?
- Open an issue on GitHub
- Check the documentation files
- Review the research plan

---

**Ready to enhance low-light images? Let's get started! 🌙→☀️**

**Project Status**: ✅ Complete & Ready to Use  
**Target Deployment**: 2026  
**License**: MIT
