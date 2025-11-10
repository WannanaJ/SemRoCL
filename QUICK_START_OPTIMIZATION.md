# 快速开始：优化实施

本指南帮助您立即开始 SemRoCL 的第一阶段优化，预计 2 周内达到初步目标。

---

## 前置准备

### 1. 检查环境

```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查现有依赖
pip list | grep -E "lpips|torch|torchvision"
```

### 2. 安装新依赖

```bash
# 感知损失相关
pip install DISTS-pytorch
pip install piq
pip install kornia

# 可视化增强
pip install tensorboard
pip install matplotlib seaborn

# 性能优化
pip install timm  # 如果需要 ViT
```

---

## 第一步：感知损失增强（3-5天）

### Day 1: 实现基础模块

1. **创建增强感知损失模块**

将以下代码添加到 [src/loss_functions.py](src/loss_functions.py) 末尾：

```python
# ========================================
# Advanced Perceptual Loss (新增)
# ========================================

import DISTS_pytorch as DISTS
import torchvision

class AdvancedPerceptualLoss(nn.Module):
    """
    多尺度感知损失模块
    - LPIPS: 0.4 weight
    - DISTS: 0.3 weight
    - Style: 0.3 weight
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # LPIPS
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()

        # DISTS
        self.dists = DISTS.DISTS().to(device).eval()

        # VGG for style loss
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleDict({
            'relu1_2': nn.Sequential(*list(vgg.children())[:4]),
            'relu2_2': nn.Sequential(*list(vgg.children())[:9]),
            'relu3_4': nn.Sequential(*list(vgg.children())[:18]),
            'relu4_4': nn.Sequential(*list(vgg.children())[:27])
        }).to(device).eval()

        # Freeze VGG
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.w_lpips = 0.4
        self.w_dists = 0.3
        self.w_style = 0.3

    def gram_matrix(self, features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def compute_style_loss(self, pred, target):
        loss = 0
        for layer_name, layer in self.vgg_layers.items():
            pred_feat = layer(pred)
            target_feat = layer(target)

            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)

            loss += F.mse_loss(pred_gram, target_gram)

        return loss / len(self.vgg_layers)

    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, 3, H, W), normalized to [-1, 1] or [0, 1]
        """
        # Ensure [0, 1] range
        pred = (pred + 1) / 2 if pred.min() < 0 else pred
        target = (target + 1) / 2 if target.min() < 0 else target

        # LPIPS
        with torch.no_grad():
            loss_lpips = self.lpips(pred, target).mean()

        # DISTS
        with torch.no_grad():
            loss_dists = self.dists(pred, target).mean()

        # Style
        loss_style = self.compute_style_loss(pred, target)

        # Weighted sum
        total_loss = (
            self.w_lpips * loss_lpips +
            self.w_dists * loss_dists +
            self.w_style * loss_style
        )

        return total_loss, {
            'lpips': loss_lpips.item(),
            'dists': loss_dists.item(),
            'style': loss_style.item(),
            'total': total_loss.item()
        }


# Helper function to use in training
def create_advanced_perceptual_loss(device='cuda'):
    """Factory function for easy initialization"""
    return AdvancedPerceptualLoss(device=device)
```

2. **测试模块**

```python
# test_perceptual_loss.py
import torch
from src.loss_functions import create_advanced_perceptual_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = create_advanced_perceptual_loss(device)

# 测试数据
pred = torch.rand(2, 3, 256, 256).to(device)
target = torch.rand(2, 3, 256, 256).to(device)

loss, components = loss_fn(pred, target)
print(f"Total Loss: {loss.item():.4f}")
print(f"Components: {components}")
```

### Day 2-3: 集成到训练

修改 [src/train_stage2_enhanced.py](src/train_stage2_enhanced.py):

```python
# 在文件开头导入
from src.loss_functions import create_advanced_perceptual_loss

# 在 main() 函数中初始化
def main():
    # ... 现有代码 ...

    # 初始化损失函数
    perceptual_loss = create_advanced_perceptual_loss(device)
    logger.info("Initialized advanced perceptual loss")

    # 训练循环
    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            low = batch['low'].to(device)
            high = batch['high'].to(device)

            # Forward
            enhanced = model(low)

            # 计算损失
            losses = {}

            # 1. 感知损失（增强版）
            losses['perceptual'], perc_components = perceptual_loss(enhanced, high)

            # 2. L1 损失（降低权重）
            losses['l1'] = F.l1_loss(enhanced, high)

            # 3. SSIM 损失
            losses['ssim'] = 1 - ssim_loss(enhanced, high)

            # 4. 其他损失（如果启用）
            if config.loss.get('semantic', {}).get('enabled', False):
                losses['semantic'] = compute_semantic_loss(model, low, high)

            if config.loss.get('frequency', {}).get('enabled', False):
                losses['freq'] = compute_frequency_loss(enhanced, high)

            # 总损失（按配置权重）
            total_loss = (
                config.loss.perceptual.weight * losses['perceptual'] +
                config.loss.reconstruction.l1_weight * losses['l1'] +
                config.loss.reconstruction.ssim_weight * losses['ssim']
            )

            if 'semantic' in losses:
                total_loss += config.loss.semantic.feature_weight * losses['semantic']

            if 'freq' in losses:
                total_loss += config.loss.frequency.fft_weight * losses['freq']

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch [{epoch}/{config.training.num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {total_loss.item():.4f}"
                )
                logger.info(f"  Components: {perc_components}")

                # TensorBoard
                if writer is not None:
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/total', total_loss.item(), step)
                    writer.add_scalar('Loss/perceptual', losses['perceptual'].item(), step)
                    writer.add_scalar('Loss/l1', losses['l1'].item(), step)
                    for name, value in perc_components.items():
                        writer.add_scalar(f'Perceptual/{name}', value, step)
```

### Day 4-5: 训练与验证

```bash
# 1. 开始训练
python src/train_stage2_enhanced.py \
    --config configs/train_stage2_optimized_v2.yaml \
    --tag "v2_phase1_perceptual" \
    --resume outputs/semantic_enhancement_stage2_ENHANCED/checkpoints/best_model.pth

# 2. 监控训练（另一个终端）
tensorboard --logdir outputs/semantic_enhancement_stage2_OPTIMIZED_V2/tensorboard

# 3. 每 10 epoch 验证一次
python src/evaluate.py \
    --checkpoint outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/epoch_010.pth \
    --test-dir data/LOL-v2/Test \
    --output-dir outputs/eval_phase1 \
    --metrics psnr ssim lpips niqe brisque delta_e hpi
```

**预期结果检查**:
- LPIPS 应该在第 20-30 epoch 开始下降
- 目标: LPIPS < 0.06 (从当前的 ~0.08)
- SSIM 应该提升 0.02~0.03

---

## 第二步：Retinex-语义融合（5-7天）

### Day 6-7: 实现 Retinex 模块

创建新文件 [src/model/retinex_semantic.py](src/model/retinex_semantic.py)，复制优化路线图中的完整代码。

### Day 8-9: 集成到主模型

修改 [src/model/models_enhanced.py](src/model/models_enhanced.py):

```python
# 在文件开头导入
from .retinex_semantic import RetinexSemanticFusion, RetinexLoss

class EnhancedLowLightModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... 现有代码 ...

        # 新增: Retinex-语义融合
        use_retinex = config.model.get('retinex_semantic', {}).get('enabled', False)
        if use_retinex:
            self.retinex_fusion = RetinexSemanticFusion(
                semantic_dim=config.model.semantic_head.out_channels
            )
            self.retinex_loss_fn = RetinexLoss()
            logger.info("Enabled Retinex-Semantic Fusion")
        else:
            self.retinex_fusion = None
            self.retinex_loss_fn = None

    def forward(self, low_img, return_components=False):
        # 1. 特征提取
        features = self.encoder_q(low_img)

        # 2. 语义特征
        semantic_feat = self.semantic_head(features)

        # 3. 增强
        if self.retinex_fusion is not None:
            # Retinex-语义增强
            enhanced, retinex_components = self.retinex_fusion(low_img, semantic_feat)
        else:
            # 原始解码器
            enhanced = self.decoder(features)
            retinex_components = None

        # 4. 频域增强（如果启用）
        if self.freq_enhance is not None:
            enhanced = self.freq_enhance(enhanced)

        if return_components:
            return enhanced, {'retinex': retinex_components}
        return enhanced

    def compute_loss(self, low_img, high_img, enhanced, components=None):
        """统一的损失计算"""
        losses = {}

        # ... 现有损失 ...

        # Retinex 物理约束
        if components is not None and 'retinex' in components:
            retinex_loss, retinex_details = self.retinex_loss_fn(
                components['retinex'], low_img
            )
            losses['retinex'] = retinex_loss
            losses['retinex_details'] = retinex_details

        return losses
```

### Day 10-12: 训练与可视化

```bash
# 训练
python src/train_stage2_enhanced.py \
    --config configs/train_stage2_optimized_v2.yaml \
    --tag "v2_phase1_retinex" \
    --resume outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/best_model.pth

# 可视化 Retinex 分解
python tools/visualize_retinex.py \
    --checkpoint outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/latest.pth \
    --input data/LOL-v2/Test/Low \
    --output outputs/retinex_vis
```

创建可视化工具 [tools/visualize_retinex.py](tools/visualize_retinex.py):

```python
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from src.model.models_enhanced import EnhancedLowLightModel
from src.utils import load_image, save_image

def visualize_retinex_decomposition(model, img_path, output_dir):
    """可视化 Retinex 分解结果"""
    # 加载图像
    low_img = load_image(img_path).unsqueeze(0).cuda()

    # 前向传播
    model.eval()
    with torch.no_grad():
        enhanced, components = model(low_img, return_components=True)

    retinex_comp = components['retinex']

    # 准备可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def to_numpy(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Row 1
    axes[0, 0].imshow(to_numpy(low_img))
    axes[0, 0].set_title('Input (Low-light)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(to_numpy(retinex_comp['R']))
    axes[0, 1].set_title('Reflectance (R)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(to_numpy(retinex_comp['L']).squeeze(), cmap='gray')
    axes[0, 2].set_title('Illumination (L)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(to_numpy(retinex_comp['R_enhanced']))
    axes[1, 0].set_title('Enhanced Reflectance', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(to_numpy(retinex_comp['L_corrected']).squeeze(), cmap='gray')
    axes[1, 1].set_title('Corrected Illumination', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(to_numpy(enhanced))
    axes[1, 2].set_title('Final Output', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # 保存
    output_path = output_dir / f"{Path(img_path).stem}_retinex.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='Image path or directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # 加载模型
    model = EnhancedLowLightModel.load_from_checkpoint(args.checkpoint)
    model.cuda().eval()

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理图像
    input_path = Path(args.input)
    if input_path.is_file():
        visualize_retinex_decomposition(model, str(input_path), output_dir)
    else:
        for img_file in input_path.glob('*.png'):
            visualize_retinex_decomposition(model, str(img_file), output_dir)

if __name__ == '__main__':
    main()
```

**预期结果检查**:
- PSNR 应该提升 0.8~1.2 dB
- Retinex 分解应该清晰（R 保留结构，L 平滑）
- 边缘应该更清晰（语义引导效果）

---

## 阶段验证：跨数据集评估

完成第一阶段后，运行完整评估：

```bash
# 1. 单数据集详细评估
python src/evaluate.py \
    --checkpoint outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/best_model.pth \
    --test-dir data/LOL-v2/Test \
    --metrics psnr ssim lpips niqe brisque delta_e hpi \
    --save-images \
    --output-dir outputs/eval_phase1_final

# 2. 多数据集评估（使用实验框架）
python src/experiments/run_ablation.py \
    --config configs/experiment_auto.yaml \
    --checkpoint outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/best_model.pth

# 3. 生成报告
python src/experiments/summarize_results.py \
    --config configs/experiment_auto.yaml

python src/experiments/summarize_cross_dataset.py \
    --base-dir outputs/ablation_multi_dataset
```

---

## 成功标准检查表

### 第一阶段目标 (2周后)

- [ ] **PSNR ≥ 24 dB** (当前 + 1.5~2 dB)
- [ ] **SSIM ≥ 0.86** (当前 + 0.02~0.03)
- [ ] **LPIPS ≤ 0.06** (当前 - 0.02~0.03)
- [ ] **NIQE ≤ 3.8** (保持或略有改善)
- [ ] **HPI ≥ 0.88** (综合提升)

### 代码质量

- [ ] 所有新模块有文档字符串
- [ ] 训练可复现（固定随机种子）
- [ ] 检查点包含完整配置
- [ ] 可视化工具正常工作

### 实验记录

- [ ] TensorBoard 日志完整
- [ ] 保存了关键 epoch 的检查点
- [ ] 有可视化对比图
- [ ] 记录了训练时间和资源使用

---

## 故障排查

### 问题 1: LPIPS 不下降

**可能原因**:
- 感知损失权重过低
- VGG/LPIPS 模型未正确加载

**解决方案**:
```python
# 检查模型加载
print(perceptual_loss.lpips)
print(perceptual_loss.vgg_layers)

# 提高权重
config.loss.perceptual.weight = 0.35  # 从 0.30 提高
```

### 问题 2: 训练爆炸/NaN

**解决方案**:
```python
# 添加损失检查
if torch.isnan(total_loss) or torch.isinf(total_loss):
    logger.warning("NaN/Inf loss detected, skipping batch")
    continue

# 降低学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # 从 1e-4 降低
```

### 问题 3: 显存不足

**解决方案**:
```yaml
# 减小 batch size
training:
  batch_size: 4  # 从 8 减半
  gradient:
    accumulation_steps: 4  # 保持等效 batch size
```

---

## 下一步

完成第一阶段后，参考 [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) 进行第二阶段优化：
1. 曲线-Transformer 混合
2. ViT 对比预训练升级

预计第二阶段额外提升：
- PSNR: +1.0~1.5 dB
- 最终达到论文亮点目标
