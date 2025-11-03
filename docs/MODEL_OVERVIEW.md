# SemRoCL: Semantic-guided Robust Contrastive Learning for Low-Light Image Enhancement

## 1. Overall Framework

SemRoCL consists of **two cooperative stages**:

| Stage | Name | Objective | Learning Paradigm |
|--------|------|------------|-------------------|
| **Stage 1** | Momentum Contrastive Pretraining (MoCo v3 Encoder) | Learn robust low-light feature representations | Self-supervised contrastive learning |
| **Stage 2** | Semantic-Guided Enhancement | Enhance low-light images using semantic cues and curve-based illumination correction | Multi-objective supervised learning |

**Processing Flow:**
Low-Light Image
│
▼
[Stage 1: MoCo Encoder]
└─► Extract robust latent representation
▼
[Stage 2: Semantic-Guided Enhancer]
├─► Semantic Head (SegFormer-based)
├─► Curve Enhancer (Iterative illumination mapping)
├─► PatchGAN Discriminator (realism constraint)
└─► Multi-loss joint optimization (Color, Semantic, Frequency, Perceptual, Adversarial)
▼
Enhanced Image Output

---

## 2. Stage 1: Momentum Contrastive Pretraining

- **Goal:** Obtain illumination-invariant, structure-preserving representations under self-supervision.  
- **Architecture:** MoCo v3 encoder (ViT or ResNet backbone).  
- **Key Mechanisms:**
  - Query-Key encoder pairs with momentum update.
  - Contrastive queue for stable negative sampling.
  - Augmentations tailored to low-light domains (gamma, noise, blur).

**Output:** Feature embeddings later reused for semantic enhancement initialization.

---

## 3. Stage 2: Semantic-Guided Enhancement

- **Input:** Low-light image + pretrained encoder features.
- **Semantic Head:** SegFormer-based decoder generating pixel-level semantics & confidence maps.
- **Enhancer:** Curve-based illumination adjustment with semantic guidance.
- **Discriminator:** PatchGAN enforcing perceptual realism.
- **Loss Fusion:** Multi-objective loss balancing structure, texture, and semantics.

**Training Flow:**
Low-Light → Encoder → Semantic Head → Semantic Features
↓
Curve Enhancer → Enhanced Output
↓
Discriminator + Loss Fusion

---

## 4. Data Flow and Feature Shapes (Example: 256×256 input)

| Component | Input | Output | Description |
|------------|--------|--------|-------------|
| MoCo Encoder | (B, 3, 256, 256) | (B, 1024, 32, 32) | Latent low-light feature |
| Semantic Head | (B, 3, 256, 256) | (B, 19, 256, 256) | Semantic logits |
| Enhancer | (B, 3, 256, 256) + (B, 128, 256, 256) | (B, 3, 256, 256) | Curve-based enhancement |
| Discriminator | (B, 3, 256, 256) | (B, 1, 30, 30) | Patch-level realism score |

---

## 5. Training Phases

| Phase | Component | Training Mode |
|--------|------------|---------------|
| 1 | MoCo Encoder | Pretrained (frozen in Stage 2) |
| 2 | Semantic Head | Trainable |
| 3 | Curve Enhancer | Trainable |
| 4 | Discriminator | Alternating training |
| 5 | Multi-loss optimization | Joint tuning with adaptive weights |

---

## 6. Key Advantages

- **Semantic priors** prevent texture washout in dark regions.  
- **Curve-based mapping** ensures natural tone restoration.  
- **Contrastive pretraining** enhances stability under extreme illumination.  
- **Multi-loss optimization** balances fidelity, structure, and realism.


