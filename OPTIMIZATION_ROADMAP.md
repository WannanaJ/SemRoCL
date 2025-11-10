# SemRoCL 优化路线图
# 目标：PSNR ≥ 25 dB, SSIM ≥ 0.89, LPIPS ≤ 0.05, HPI ≥ 0.92

## 执行摘要

基于当前实验框架和目标指标，本路线图提供分阶段优化方案，预计整体提升：
- **PSNR**: +2.5~4 dB
- **SSIM**: +0.04~0.06
- **LPIPS**: -0.03~0.05
- **HPI**: +0.05~0.08

---

## 第一阶段：核心优化（2周）

### 优先级 P0：感知损失增强

**实施时间**: 3-5天
**预期提升**: LPIPS ↓ 0.02~0.03, SSIM ↑ 0.02~0.03

#### 实现步骤

1. **安装依赖**
```bash
pip install DISTS-pytorch
pip install piq
```

2. **创建增强感知损失模块**

文件位置: [src/loss_functions.py](src/loss_functions.py)

```python
import DISTS_pytorch as DISTS
import piq

class AdvancedPerceptualLoss(nn.Module):
    """
    多尺度感知损失，融合：
    - LPIPS (Alex net): 感知相似度
    - DISTS: 结构与纹理相似度
    - Style Loss: VGG 风格损失
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # 1. LPIPS（已有）
        self.lpips = lpips.LPIPS(net='alex').to(device)

        # 2. DISTS（新增）
        self.dists = DISTS.DISTS().to(device)

        # 3. Style Loss（VGG 多层特征）
        self.vgg = VGG19FeatureExtractor(
            layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']
        ).to(device)

        # 权重
        self.w_lpips = 0.4
        self.w_dists = 0.3
        self.w_style = 0.3

    def gram_matrix(self, features):
        """计算 Gram 矩阵"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def style_loss(self, pred, target):
        """风格损失"""
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)

        loss = 0
        for p_feat, t_feat in zip(pred_feats, target_feats):
            p_gram = self.gram_matrix(p_feat)
            t_gram = self.gram_matrix(t_feat)
            loss += F.mse_loss(p_gram, t_gram)

        return loss / len(pred_feats)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3, H, W), range [0, 1]
            target: (B, 3, H, W), range [0, 1]
        """
        # 1. LPIPS
        loss_lpips = self.lpips(pred, target).mean()

        # 2. DISTS
        loss_dists = self.dists(pred, target).mean()

        # 3. Style
        loss_style = self.style_loss(pred, target)

        # 加权组合
        total_loss = (
            self.w_lpips * loss_lpips +
            self.w_dists * loss_dists +
            self.w_style * loss_style
        )

        return total_loss, {
            'lpips': loss_lpips.item(),
            'dists': loss_dists.item(),
            'style': loss_style.item()
        }


class VGG19FeatureExtractor(nn.Module):
    """VGG19 特征提取器"""
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layer_name_mapping = {
            'relu1_2': '3',
            'relu2_2': '8',
            'relu3_4': '17',
            'relu4_4': '26'
        }

        self.layers = layers
        self.feature_extractor = nn.ModuleDict()

        for layer_name in layers:
            layer_idx = int(self.layer_name_mapping[layer_name])
            self.feature_extractor[layer_name] = nn.Sequential(
                *list(vgg.children())[:layer_idx+1]
            )

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer_name in self.layers:
            feat = self.feature_extractor[layer_name](x)
            features.append(feat)
        return features
```

3. **集成到训练循环**

修改 [src/train_stage2_enhanced.py](src/train_stage2_enhanced.py):

```python
# 初始化损失
perceptual_loss = AdvancedPerceptualLoss(device=device)

# 训练循环中
for batch in train_loader:
    low, high = batch['low'].to(device), batch['high'].to(device)

    # 前向
    enhanced = model(low)

    # 计算损失
    loss_perc, perc_components = perceptual_loss(enhanced, high)
    loss_l1 = F.l1_loss(enhanced, high)

    # 总损失
    loss = 0.30 * loss_perc + 0.15 * loss_l1 + other_losses

    # 日志
    if step % 50 == 0:
        logger.info(f"Perceptual: {perc_components}")
```

4. **验证效果**

```bash
# 训练
python src/train_stage2_enhanced.py \
    --config configs/train_stage2_optimized_v2.yaml \
    --tag "v2_advanced_perceptual"

# 评估
python src/evaluate.py \
    --checkpoint outputs/semantic_enhancement_stage2_OPTIMIZED_V2/checkpoints/best_model.pth \
    --test-dir data/LOL-v2/Test \
    --metrics psnr ssim lpips
```

**预期结果**:
- LPIPS: 当前值 → 目标 < 0.05
- SSIM: ↑ 0.02~0.03
- Training time: +10~15%（可接受）

---

### 优先级 P0：Retinex-语义融合

**实施时间**: 5-7天
**预期提升**: PSNR ↑ 0.8~1.2 dB, 可解释性提升

#### 实现步骤

1. **创建 Retinex 分解模块**

文件位置: [src/model/retinex_semantic.py](src/model/retinex_semantic.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexDecomposition(nn.Module):
    """
    Retinex 分解：I = R * L
    - R: Reflectance (反射率，物体固有属性)
    - L: Illumination (光照，环境因素)
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # 使用可学习的分解网络（基于 DecomNet）
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

        self.illumination_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),  # 单通道光照图
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W), range [0, 1]
        Returns:
            R: (B, 3, H, W), reflectance
            L: (B, 1, H, W), illumination
        """
        R = self.reflectance_net(img)
        L = self.illumination_net(img)

        return R, L


class SemanticGuidedRefine(nn.Module):
    """语义引导的反射率增强"""
    def __init__(self, semantic_dim=512, out_channels=3):
        super().__init__()

        # 语义特征映射
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1)
        )

        # 反射率细化（融合语义）
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + 64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()  # 输出残差
        )

    def forward(self, R, semantic_feat):
        """
        Args:
            R: (B, 3, H, W), 原始反射率
            semantic_feat: (B, C, H, W), 语义特征
        """
        # 调整语义特征尺寸
        if semantic_feat.shape[2:] != R.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=R.shape[2:],
                mode='bilinear', align_corners=False
            )

        # 映射语义特征
        sem = self.semantic_proj(semantic_feat)

        # 融合并细化
        fused = torch.cat([R, sem], dim=1)
        residual = self.refine(fused)

        # 残差连接
        R_enhanced = torch.clamp(R + 0.2 * residual, 0, 1)

        return R_enhanced


class SemanticGuidedIllumination(nn.Module):
    """语义引导的光照校正"""
    def __init__(self, semantic_dim=512):
        super().__init__()

        # 语义到光照映射
        self.illumination_adjust = nn.Sequential(
            nn.Conv2d(1 + semantic_dim, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, L, semantic_feat):
        """
        Args:
            L: (B, 1, H, W), 原始光照图
            semantic_feat: (B, C, H, W), 语义特征
        """
        # 调整尺寸
        if semantic_feat.shape[2:] != L.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=L.shape[2:],
                mode='bilinear', align_corners=False
            )

        # 融合
        fused = torch.cat([L, semantic_feat], dim=1)

        # 光照校正
        L_corrected = self.illumination_adjust(fused)

        return L_corrected


class RetinexSemanticFusion(nn.Module):
    """Retinex-语义融合主模块"""
    def __init__(self, semantic_dim=512):
        super().__init__()

        self.decomposition = RetinexDecomposition()
        self.reflectance_enhance = SemanticGuidedRefine(semantic_dim)
        self.illumination_adjust = SemanticGuidedIllumination(semantic_dim)

    def forward(self, low_img, semantic_feat):
        """
        Args:
            low_img: (B, 3, H, W), 低光图像
            semantic_feat: (B, C, H, W), 语义特征

        Returns:
            enhanced: (B, 3, H, W), 增强图像
            components: dict, 中间结果
        """
        # 1. Retinex 分解
        R, L = self.decomposition(low_img)

        # 2. 语义引导反射率增强
        R_enhanced = self.reflectance_enhance(R, semantic_feat)

        # 3. 语义引导光照校正
        L_corrected = self.illumination_adjust(L, semantic_feat)

        # 4. 重建
        enhanced = R_enhanced * L_corrected.expand_as(R_enhanced)

        return enhanced, {
            'R': R,
            'L': L,
            'R_enhanced': R_enhanced,
            'L_corrected': L_corrected
        }


class RetinexLoss(nn.Module):
    """Retinex 物理约束损失"""
    def __init__(self):
        super().__init__()

    def illumination_smoothness_loss(self, L):
        """光照平滑损失（TV loss）"""
        loss_h = torch.mean(torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :]))
        loss_w = torch.mean(torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1]))
        return loss_h + loss_w

    def reflectance_consistency_loss(self, R, img):
        """反射率一致性（保留结构）"""
        # 使用梯度一致性
        grad_R_h = torch.abs(R[:, :, 1:, :] - R[:, :, :-1, :])
        grad_img_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        loss_h = F.l1_loss(grad_R_h, grad_img_h)

        grad_R_w = torch.abs(R[:, :, :, 1:] - R[:, :, :, :-1])
        grad_img_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        loss_w = F.l1_loss(grad_R_w, grad_img_w)

        return loss_h + loss_w

    def reconstruction_loss(self, R, L, img):
        """重建损失：I = R * L"""
        reconstructed = R * L.expand_as(R)
        return F.l1_loss(reconstructed, img)

    def forward(self, components, low_img):
        """
        Args:
            components: dict from RetinexSemanticFusion
            low_img: (B, 3, H, W)
        """
        R = components['R']
        L = components['L']

        loss_smooth = self.illumination_smoothness_loss(L)
        loss_consist = self.reflectance_consistency_loss(R, low_img)
        loss_recon = self.reconstruction_loss(R, L, low_img)

        total = 0.08 * loss_smooth + 0.05 * loss_consist + 0.05 * loss_recon

        return total, {
            'smooth': loss_smooth.item(),
            'consist': loss_consist.item(),
            'recon': loss_recon.item()
        }
```

2. **集成到主模型**

修改 [src/model/models_enhanced.py](src/model/models_enhanced.py):

```python
from .retinex_semantic import RetinexSemanticFusion, RetinexLoss

class EnhancedLowLightModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... 现有代码 ...

        # 添加 Retinex-语义融合
        if config.model.get('retinex_semantic', {}).get('enabled', False):
            self.retinex_semantic = RetinexSemanticFusion(
                semantic_dim=config.model.semantic_head.out_channels
            )
            self.retinex_loss = RetinexLoss()
        else:
            self.retinex_semantic = None

    def forward(self, low_img, return_components=False):
        # 1. 编码器提取特征
        features = self.encoder_q(low_img)

        # 2. 语义特征
        semantic_feat = self.semantic_head(features)

        # 3. Retinex-语义增强
        if self.retinex_semantic is not None:
            enhanced, retinex_components = self.retinex_semantic(
                low_img, semantic_feat
            )
        else:
            # 原始解码器
            enhanced = self.decoder(features)
            retinex_components = None

        # 4. 频域增强（如果启用）
        if self.freq_enhance is not None:
            enhanced = self.freq_enhance(enhanced)

        if return_components:
            return enhanced, retinex_components
        return enhanced
```

3. **更新训练脚本**

修改 [src/train_stage2_enhanced.py](src/train_stage2_enhanced.py):

```python
# 训练循环
for epoch in range(start_epoch, config.training.num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        low, high = batch['low'].to(device), batch['high'].to(device)

        # 前向
        enhanced, retinex_components = model(low, return_components=True)

        # 损失计算
        losses = {}

        # 1. 重建损失
        losses['l1'] = F.l1_loss(enhanced, high)

        # 2. 感知损失
        losses['perceptual'], _ = perceptual_loss(enhanced, high)

        # 3. Retinex 物理约束
        if retinex_components is not None:
            losses['retinex'], retinex_details = model.retinex_loss(
                retinex_components, low
            )

        # 4. 其他损失...

        # 总损失
        total_loss = (
            0.15 * losses['l1'] +
            0.30 * losses['perceptual'] +
            0.08 * losses.get('retinex', 0) +
            # ... 其他损失
        )

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 日志
        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                       f"Loss: {total_loss:.4f} | L1: {losses['l1']:.4f} | "
                       f"Perc: {losses['perceptual']:.4f} | "
                       f"Retinex: {losses.get('retinex', 0):.4f}")

            if retinex_components is not None:
                logger.info(f"  Retinex details: {retinex_details}")
```

4. **可视化 Retinex 分解**

```python
# 在验证时保存中间结果
def visualize_retinex(low_img, enhanced, components, save_path):
    """可视化 Retinex 分解"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原始图像
    axes[0, 0].imshow(tensor_to_numpy(low_img))
    axes[0, 0].set_title('Low-light Input')

    # 反射率
    axes[0, 1].imshow(tensor_to_numpy(components['R']))
    axes[0, 1].set_title('Reflectance (R)')

    # 光照
    axes[0, 2].imshow(tensor_to_numpy(components['L']), cmap='gray')
    axes[0, 2].set_title('Illumination (L)')

    # 增强反射率
    axes[1, 0].imshow(tensor_to_numpy(components['R_enhanced']))
    axes[1, 0].set_title('Enhanced Reflectance')

    # 校正光照
    axes[1, 1].imshow(tensor_to_numpy(components['L_corrected']), cmap='gray')
    axes[1, 1].set_title('Corrected Illumination')

    # 最终结果
    axes[1, 2].imshow(tensor_to_numpy(enhanced))
    axes[1, 2].set_title('Enhanced Output')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**预期结果**:
- PSNR: ↑ 0.8~1.2 dB
- 可解释性：可视化 R/L 分解，论文中可展示
- 边缘保持：语义引导避免过度平滑

---

## 第二阶段：进阶优化（3-4周）

### 优先级 P1：曲线-Transformer 混合

**实施时间**: 7-10天
**预期提升**: PSNR ↑ 1.0~1.5 dB, 参数可解释

详细实现见配置文件和后续文档。

### 优先级 P1：对比预训练升级（ViT）

**实施时间**: 1-2周（需要重新预训练）
**预期提升**: PSNR ↑ 0.4~0.6 dB

仅在时间允许的情况下实施。

---

## 第三阶段：精调与验证（1-2周）

### 跨数据集评估

使用已建立的实验框架：

```bash
# 1. 扫描数据集
python src/experiments/scan_datasets.py --base-dir data/stage3_eval

# 2. 运行多数据集实验
bash scripts/run_multi_dataset_experiments.bat

# 3. 汇总结果
python src/experiments/summarize_cross_dataset.py \
    --base-dir outputs/ablation_multi_dataset
```

### 性能优化

1. **模型剪枝**（如果参数量 > 10M）
```python
import torch_pruning as tp

# 剪枝 20% 的通道
pruner = tp.pruner.MagnitudePruner(
    model, example_inputs, importance=tp.importance.MagnitudeImportance(),
    pruning_ratio=0.2
)
pruner.step()
```

2. **量化加速**
```python
# 动态量化
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
```

3. **TensorRT 优化**（如果需要实时推理）

---

## 关键里程碑

| 时间点 | 里程碑 | 目标指标 |
|--------|--------|----------|
| Week 1 | 感知损失增强完成 | LPIPS < 0.06 |
| Week 2 | Retinex-语义融合完成 | PSNR ≥ 24 dB |
| Week 4 | 第一阶段验证 | 达到稳定领先下限 |
| Week 6 | 曲线-Transformer完成 | PSNR ≥ 25 dB |
| Week 8 | 最终优化与跨数据集验证 | 所有指标达标 |

---

## 风险管理

### 风险 1：训练不稳定

**缓解措施**:
- 使用梯度裁剪 (norm=1.0)
- Warmup 学习率 (20 epochs)
- 混合精度训练 + 损失缩放

### 风险 2：过拟合

**缓解措施**:
- 数据增强 (flip, rotation, color jitter)
- Dropout / DropPath
- 早停 (patience=50)

### 风险 3：计算资源不足

**缓解措施**:
- 梯度累积 (accumulation_steps=2)
- 混合精度训练 (节省 40% 显存)
- 检查点保存策略 (只保留 best 5)

---

## 成功标准

### 最低标准（必须达到）
- PSNR ≥ 24 dB
- SSIM ≥ 0.86
- LPIPS ≤ 0.07
- HPI ≥ 0.90

### 目标标准（论文亮点）
- PSNR ≥ 25 dB
- SSIM ≥ 0.89
- LPIPS ≤ 0.05
- HPI ≥ 0.92
- 参数量 < 10M
- 推理速度 < 30 ms @ 512²

---

## 资源需求

### 硬件
- GPU: RTX 3090 或更高
- 内存: 32GB+
- 存储: 500GB+ (数据集 + 检查点)

### 软件
- PyTorch >= 1.12
- CUDA >= 11.3
- 新增依赖: DISTS-pytorch, piq

### 时间
- 开发时间: 4-6周
- 训练时间: 每个实验 10-15 GPU小时
- 总计: 约 200 GPU小时

---

## 后续工作

完成上述优化后，可考虑：

1. **视频低光增强**（频域-时域协同）
2. **多尺度语义蒸馏**
3. **实时推理优化** (TensorRT / ONNX)
4. **开源与论文投稿**
