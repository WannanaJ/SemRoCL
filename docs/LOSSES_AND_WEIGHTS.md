# SemRoCL Multi-Loss Design and Weighting Strategy

## 1. Overview

The Stage 2 training optimizes a **composite objective** combining five primary losses:

\[
\mathcal{L}_{total} = \lambda_c \mathcal{L}_{color} + \lambda_s \mathcal{L}_{semantic} + \lambda_f \mathcal{L}_{freq} + \lambda_p \mathcal{L}_{perceptual} + \lambda_{adv} \mathcal{L}_{adv}
\]

Each component targets a complementary aspect of enhancement quality.

---

## 2. Loss Definitions

| Loss | Formula / Description | Target | Typical Weight (λ) | Notes |
|------|------------------------|---------|--------------------|-------|
| **Color Loss** | L1 distance between enhanced and ground truth RGB | Pixel-wise fidelity | **1.0** | Stabilizes base color recovery |
| **Semantic Loss** | Cross-entropy between enhanced and target segmentation logits | Semantic consistency | **0.5** | Weighted by confidence map |
| **Frequency Loss** | L1 loss in FFT domain | Texture detail & sharpness | **0.1** | Reduces blur, enhances edges |
| **Perceptual (LPIPS) Loss** | LPIPS distance using VGG backbone | Human perceptual alignment | **0.2** | Improves visual realism |
| **Adversarial Loss** | PatchGAN discriminator loss | Realism / naturalness | **0.005** | Enabled after warm-up epochs |

---

## 3. Dynamic Weight Scheduling (Optional)

| Strategy | Description | Benefit |
|-----------|--------------|----------|
| **Warm-up Adversarial** | Gradually increase λ_adv from 0→0.005 over first 10 epochs | Prevent instability |
| **Cosine Decay on Frequency Loss** | Decay λ_f after 100 epochs | Reduce over-sharpening |
| **Confidence-Gated Semantic Weight** | λ_s × mean(confidence) | Adaptive semantic reliability |
| **Loss Normalization** | Normalize all λ_i based on moving average magnitudes | Automatic balancing |

---

## 4. Additional Losses (optional extensions)

| Loss | Formula | Motivation |
|------|----------|------------|
| **Task Loss** | (e.g., L1 between enhanced and reference brightness map) | Regularize enhancement magnitude |
| **Edge Loss** | L1 on gradient maps | Preserve structure boundaries |
| **SSIM Loss** | (1 − SSIM) | Complementary perceptual alignment |

---

## 5. Recommended Weight Configuration (Baseline)

```yaml
loss:
  color_weight: 1.0
  semantic_weight: 0.5
  freq_weight: 0.1
  perceptual_weight: 0.2
  adv_weight: 0.005
  task_weight: 0.2
6. Stability Guidelines

Clamp LPIPS and frequency losses to avoid NaNs.

Use gradient clipping (max_norm=1.0).

Monitor relative scale of each loss in TensorBoard.

Apply AMP (torch.amp.autocast('cuda')) for stability and speed.

---

