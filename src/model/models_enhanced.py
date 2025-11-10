"""
Enhanced Model Components for SemRoCL Stage 2
Adds lightweight controls for multi-scale enhancement and semantic guidance.
"""

from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for lightweight generator mode."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EnhancementGenerator(nn.Module):
    """
    Curve-based enhancer with optional multi-scale branches and semantic fusion.
    Lightweight mode uses channel scaling plus depthwise separable convolutions.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}

        base_channels = config.get('enhancer_channels', 32)
        self.channel_multiplier = float(config.get('channel_multiplier', 1.0))
        self.n_channels = max(16, int(base_channels * self.channel_multiplier))

        self.num_iterations = int(config.get('num_iterations', 8))
        self.use_semantic = bool(config.get('use_semantic', True))
        self.semantic_channels = int(config.get('semantic_channels', 64))
        self.use_multiscale = bool(config.get('use_multiscale', False))
        self.num_scales = max(1, int(config.get('num_scales', 3)))
        self.use_noise_gate = bool(config.get('use_noise_gate', True))
        self.noise_gate_kernel = int(config.get('noise_gate_kernel', 3))
        self.lightweight = bool(config.get('lightweight', False))
        self.use_depthwise = bool(config.get('use_depthwise', self.lightweight))

        if self.use_multiscale:
            self._build_multiscale_enhancer()
        else:
            self._build_single_scale_enhancer()

        if self.use_noise_gate and self.use_multiscale:
            self._build_noise_gate()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def _make_conv(self, in_channels: int, out_channels: int,
                   kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Module:
        if self.use_depthwise:
            return DepthwiseSeparableConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

    def _build_single_scale_enhancer(self) -> None:
        self.conv1 = self._make_conv(3, self.n_channels)
        self.conv2 = self._make_conv(self.n_channels, self.n_channels)
        self.conv3 = self._make_conv(self.n_channels, self.n_channels)
        self.conv4 = self._make_conv(self.n_channels, self.n_channels)

        if self.use_semantic:
            self.semantic_fusion = nn.Sequential(
                nn.Conv2d(self.n_channels + self.semantic_channels, self.n_channels, 1),
                nn.ReLU(inplace=True),
            )
        else:
            self.semantic_fusion = None

        self.conv5 = nn.Conv2d(self.n_channels, 3 * self.num_iterations, 3, 1, 1, bias=True)

    def _build_multiscale_enhancer(self) -> None:
        self.scale_enhancers = nn.ModuleList()
        for _ in range(self.num_scales):
            scale_net = nn.ModuleDict({
                'conv1': self._make_conv(3, self.n_channels),
                'conv2': self._make_conv(self.n_channels, self.n_channels),
                'conv3': self._make_conv(self.n_channels, self.n_channels),
                'conv4': self._make_conv(self.n_channels, self.n_channels),
                'conv5': nn.Conv2d(self.n_channels, 3 * self.num_iterations, 3, 1, 1, bias=True),
            })

            if self.use_semantic:
                scale_net['semantic_fusion'] = nn.Sequential(
                    nn.Conv2d(self.n_channels + self.semantic_channels, self.n_channels, 1),
                    nn.ReLU(inplace=True),
                )

            self.scale_enhancers.append(scale_net)

        fusion_hidden = max(32, int(64 * self.channel_multiplier))
        fusion_mid = max(32, self.n_channels // 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * self.num_scales, fusion_hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden, fusion_mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_mid, 3, 3, 1, 1),
        )
        output_conv = cast(nn.Conv2d, self.fusion[-1])
        nn.init.zeros_(output_conv.weight)
        if output_conv.bias is not None:
            nn.init.zeros_(output_conv.bias)

    def _build_noise_gate(self) -> None:
        kernel = max(3, self.noise_gate_kernel | 1)  # ensure odd >=3
        padding = kernel // 2
        self.noise_smoother = nn.Conv2d(3, 3, kernel_size=kernel, padding=padding, groups=3, bias=False)
        with torch.no_grad():
            self.noise_smoother.weight.zero_()
            self.noise_smoother.weight[:, :, padding, padding] = 1.0

        self.noise_gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                                   [1.0, -4.0, 1.0],
                                   [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', lap_kernel.view(1, 1, 3, 3))

    def _semantic_fuse_block(
        self,
        fusion_module: Optional[nn.Module],
        features: torch.Tensor,
        semantic_features: Optional[torch.Tensor],
        confidence_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if fusion_module is None or semantic_features is None or not self.use_semantic:
            return features

        semantic_resized = F.interpolate(
            semantic_features,
            size=features.shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        fused = torch.cat([features, semantic_resized], dim=1)
        fused = fusion_module(fused)

        if confidence_map is not None:
            conf = F.interpolate(confidence_map, size=fused.shape[2:], mode='bilinear', align_corners=False)
            fused = fused * conf
        return fused

    def _apply_noise_gate(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        variance = torch.mean((x - local_mean) ** 2, dim=1, keepdim=True)
        norm_var = variance / (variance.mean(dim=[2, 3], keepdim=True) + 1e-6)
        lap = F.conv2d(
            x,
            self.laplacian_kernel.expand(x.size(1), 1, 3, 3),
            padding=1,
            groups=x.size(1),
        )
        lap = lap.abs().mean(dim=1, keepdim=True)
        noise_feat = torch.cat([norm_var, lap], dim=1)
        gate = self.noise_gate(noise_feat)
        smoothed_residual = self.noise_smoother(residual)
        return residual * (1.0 - gate) + smoothed_residual * gate

    def _enhance_single_scale(
        self,
        x: torch.Tensor,
        semantic_features: Optional[torch.Tensor],
        confidence_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x4 = self._semantic_fuse_block(self.semantic_fusion, x4, semantic_features, confidence_map)

        curves = self.tanh(self.conv5(x4))
        enhanced = x
        for i in range(self.num_iterations):
            curve_params = curves[:, i * 3:(i + 1) * 3, :, :]
            enhanced = enhanced + curve_params * enhanced * (1 - enhanced)
            enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced

    def _enhance_multiscale(
        self,
        x: torch.Tensor,
        semantic_features: Optional[torch.Tensor],
        confidence_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        enhanced_scales = []

        for scale_idx, scale_net in enumerate(self.scale_enhancers):
            scale_net = cast(nn.ModuleDict, scale_net)
            scale = 2 ** scale_idx
            x_scaled = F.interpolate(x, scale_factor=1 / scale, mode='bilinear', align_corners=False)

            x1 = self.relu(scale_net['conv1'](x_scaled))
            x2 = self.relu(scale_net['conv2'](x1))
            x3 = self.relu(scale_net['conv3'](x2))
            x4 = self.relu(scale_net['conv4'](x3))

            fusion_module = scale_net['semantic_fusion'] if 'semantic_fusion' in scale_net else None
            x4 = self._semantic_fuse_block(fusion_module, x4, semantic_features, confidence_map)

            curves = self.tanh(scale_net['conv5'](x4))
            enhanced_scaled = x_scaled
            for i in range(self.num_iterations):
                curve_params = curves[:, i * 3:(i + 1) * 3, :, :]
                enhanced_scaled = enhanced_scaled + curve_params * enhanced_scaled * (1 - enhanced_scaled)
                enhanced_scaled = torch.clamp(enhanced_scaled, 0, 1)

            enhanced_upsampled = F.interpolate(enhanced_scaled, size=(H, W), mode='bilinear', align_corners=False)
            enhanced_scales.append(enhanced_upsampled)

        multi_scale_features = torch.cat(enhanced_scales, dim=1)
        residual = self.fusion(multi_scale_features)
        if self.use_noise_gate:
            residual = self._apply_noise_gate(x, residual)

        return torch.clamp(x + residual, 0, 1)

    def forward(
        self,
        x: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        confidence_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_multiscale:
            return self._enhance_multiscale(x, semantic_features, confidence_map)
        return self._enhance_single_scale(x, semantic_features, confidence_map)


class SemanticGuidanceModule(nn.Module):
    """
    SegFormer-based semantic guidance with optional input downsampling for efficiency.
    Returns segmentation logits, confidence map, and projected semantic features.
    """

    def __init__(
        self,
        num_classes: int = 19,
        pretrained: bool = True,
        output_features: bool = True,
        downsample_ratio: int = 1,
        feature_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_features = output_features
        self.downsample_ratio = max(1, int(downsample_ratio))
        self.output_channels = feature_dim

        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        if pretrained:
            self.segformer = cast(
                SegformerForSemanticSegmentation,
                SegformerForSemanticSegmentation.from_pretrained(
                    model_id,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                ),
            )
        else:
            from transformers import SegformerConfig

            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.segformer = SegformerForSemanticSegmentation(config)

        mid_channels = max(32, feature_dim * 2)
        self.feature_projector = nn.Sequential(
            nn.Conv2d(160, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, feature_dim, 1),
            nn.ReLU(inplace=True),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _, _, H, W = x.shape
        if self.downsample_ratio > 1:
            seg_input = F.interpolate(
                x,
                scale_factor=1.0 / self.downsample_ratio,
                mode='bilinear',
                align_corners=False,
            )
        else:
            seg_input = x

        outputs = cast(
            Any,
            self.segformer(
                pixel_values=seg_input,
                output_hidden_states=True,
                return_dict=True,
            ),
        )

        seg_logits = outputs.logits
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        confidence = self.uncertainty_head(seg_logits)

        features: Optional[torch.Tensor] = None
        if self.output_features:
            hidden_states = outputs.hidden_states
            features = hidden_states[2]
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
            features = self.feature_projector(features)

        return seg_logits, confidence, features


class Discriminator(nn.Module):
    """PatchGAN discriminator with optional spectral normalization."""

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm

        layers = []
        first_conv = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        layers.append(nn.utils.spectral_norm(first_conv) if use_spectral_norm else first_conv)
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        for _ in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 * nf_mult, 8)
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(conv))
            else:
                layers.extend([conv, nn.BatchNorm2d(ndf * nf_mult)])
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult_prev = nf_mult
        nf_mult = min(nf_mult * 2, 8)
        conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(conv))
        else:
            layers.extend([conv, nn.BatchNorm2d(ndf * nf_mult)])
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OptimizedEnhancementModel(nn.Module):
    """
    Complete Optimized Enhancement Model for TIP Submission
    Integrates P0 + P1 optimizations:

    P0 Optimizations:
    - Retinex-Semantic Fusion (physical model + deep learning)
    - Advanced Perceptual Loss (handled in loss_functions.py)

    P1 Optimizations:
    - Curve-Transformer hybrid enhancement

    Architecture Pipeline:
    1. Extract semantic features (SemanticGuidanceModule)
    2. Retinex decomposition with semantic guidance (RetinexSemanticFusion)
    3. Curve-Transformer enhancement (CurveTransformerEnhancer)
    4. Optional final refinement (EnhancementGenerator)

    Target Metrics:
    - PSNR ≥ 25 dB, SSIM ≥ 0.89, LPIPS ≤ 0.05, HPI ≥ 0.92
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}

        # Feature flags for ablation studies
        self.use_retinex = config.get('use_retinex', True)
        self.use_curve_transformer = config.get('use_curve_transformer', True)
        self.use_final_refine = config.get('use_final_refine', False)
        self.use_semantic = config.get('use_semantic', True)

        # Semantic feature dimension
        self.semantic_dim = config.get('semantic_feature_dim', 64)

        # 1. Semantic Guidance Module
        if self.use_semantic:
            self.semantic_module = SemanticGuidanceModule(
                num_classes=config.get('num_classes', 19),
                pretrained=config.get('semantic_pretrained', True),
                output_features=True,
                downsample_ratio=config.get('semantic_downsample', 1),
                feature_dim=self.semantic_dim
            )
        else:
            self.semantic_module = None

        # 2. P0: Retinex-Semantic Fusion
        if self.use_retinex:
            try:
                from .retinex_semantic import RetinexSemanticFusion
                self.retinex_fusion = RetinexSemanticFusion(
                    semantic_dim=self.semantic_dim
                )
            except ImportError:
                print("Warning: retinex_semantic module not found. Disabling Retinex fusion.")
                self.use_retinex = False
                self.retinex_fusion = None
        else:
            self.retinex_fusion = None

        # 3. P1: Curve-Transformer Enhancement
        if self.use_curve_transformer:
            try:
                from .curve_transformer import CurveTransformerEnhancer
                self.curve_transformer = CurveTransformerEnhancer(
                    in_channels=3,
                    transformer_embed_dim=config.get('transformer_embed_dim', 96),
                    transformer_out_dim=config.get('transformer_out_dim', 768),
                    num_curve_points=config.get('num_curve_points', 8),
                    use_residual_refine=True
                )
            except ImportError:
                print("Warning: curve_transformer module not found. Disabling Curve-Transformer.")
                self.use_curve_transformer = False
                self.curve_transformer = None
        else:
            self.curve_transformer = None

        # 4. Optional: Final refinement with original EnhancementGenerator
        if self.use_final_refine:
            refine_config = {
                'enhancer_channels': config.get('enhancer_channels', 32),
                'num_iterations': config.get('num_iterations', 8),
                'semantic_channels': self.semantic_dim,
                'use_semantic': self.use_semantic,
                'use_multiscale': False,  # Keep lightweight
                'lightweight': True,
                'channel_multiplier': 0.5,
            }
            self.final_refine = EnhancementGenerator(refine_config)
        else:
            self.final_refine = None

        print("\n" + "="*70)
        print("OptimizedEnhancementModel Configuration")
        print("="*70)
        print(f"  Retinex-Semantic Fusion (P0):  {self.use_retinex}")
        print(f"  Curve-Transformer (P1):        {self.use_curve_transformer}")
        print(f"  Semantic Guidance:             {self.use_semantic}")
        print(f"  Final Refinement:              {self.use_final_refine}")
        print(f"  Semantic Feature Dim:          {self.semantic_dim}")
        print("="*70 + "\n")

    def forward(
        self,
        low_img: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through complete optimized pipeline

        Args:
            low_img: Low-light input image (B, 3, H, W), range [0, 1]
            return_components: If True, return intermediate outputs for loss computation

        Returns:
            enhanced: Final enhanced image (B, 3, H, W)
            components: Dictionary with intermediate outputs (if return_components=True):
                - semantic_features: Semantic features (B, C, H, W)
                - semantic_logits: Segmentation logits (B, num_classes, H, W)
                - confidence: Confidence map (B, 1, H, W)
                - retinex_components: Dict with R, L, R_enhanced, L_corrected
                - curve_params: Curve parameters (B, 3, num_points)
                - retinex_output: Output after Retinex fusion
                - curve_output: Output after Curve-Transformer
        """
        components = {} if return_components else None
        enhanced = low_img

        # 1. Extract semantic features
        semantic_features = None
        if self.use_semantic and self.semantic_module is not None:
            seg_logits, confidence, semantic_features = self.semantic_module(low_img)
            if return_components:
                components['semantic_features'] = semantic_features
                components['semantic_logits'] = seg_logits
                components['confidence'] = confidence

        # 2. P0: Retinex-Semantic Fusion
        retinex_components = None
        if self.use_retinex and self.retinex_fusion is not None:
            if semantic_features is not None:
                enhanced, retinex_components = self.retinex_fusion(enhanced, semantic_features)
                if return_components:
                    components['retinex_components'] = retinex_components
                    components['retinex_output'] = enhanced
            else:
                print("Warning: Semantic features not available for Retinex fusion")

        # 3. P1: Curve-Transformer Enhancement
        curve_params = None
        if self.use_curve_transformer and self.curve_transformer is not None:
            enhanced, curve_params = self.curve_transformer(enhanced)
            if return_components:
                components['curve_params'] = curve_params
                components['curve_output'] = enhanced

        # 4. Optional: Final refinement
        if self.use_final_refine and self.final_refine is not None:
            enhanced = self.final_refine(
                enhanced,
                semantic_features=semantic_features,
                confidence_map=confidence if self.use_semantic else None
            )
            if return_components:
                components['final_output'] = enhanced

        return enhanced, components

    def get_retinex_components(self) -> Optional[dict]:
        """Get Retinex components for visualization (after forward pass)"""
        return getattr(self, '_last_retinex_components', None)

    def get_curve_params(self) -> Optional[torch.Tensor]:
        """Get curve parameters for visualization (after forward pass)"""
        return getattr(self, '_last_curve_params', None)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running enhanced model component tests...")
    print("="*70)

    # Test 1: Original EnhancementGenerator
    print("\n[Test 1] EnhancementGenerator...")
    gen_cfg = {
        'enhancer_channels': 32,
        'num_iterations': 8,
        'semantic_channels': 64,
        'use_semantic': True,
        'use_multiscale': True,
        'num_scales': 3,
        'lightweight': True,
        'channel_multiplier': 0.75,
    }
    generator = EnhancementGenerator(gen_cfg).to(device)
    dummy = torch.rand(2, 3, 256, 256).to(device)
    semantic = torch.rand(2, 64, 256, 256).to(device)
    out = generator(dummy, semantic)
    print(f"  Generator output shape: {out.shape}")

    # Test 2: SemanticGuidanceModule
    print("\n[Test 2] SemanticGuidanceModule...")
    semantic_head = SemanticGuidanceModule(
        num_classes=19,
        pretrained=False,
        downsample_ratio=2,
        feature_dim=64,
    ).to(device)
    seg_logits, confidence, features = semantic_head(dummy)
    print(f"  Semantic logits: {seg_logits.shape}")
    print(f"  Confidence: {confidence.shape}")
    print(f"  Features: {features.shape}")

    # Test 3: Discriminator
    print("\n[Test 3] Discriminator...")
    discriminator = Discriminator().to(device)
    disc_out = discriminator(out)
    print(f"  Discriminator output shape: {disc_out.shape}")

    # Test 4: OptimizedEnhancementModel
    print("\n[Test 4] OptimizedEnhancementModel (P0+P1)...")
    optimized_cfg = {
        'use_retinex': True,
        'use_curve_transformer': True,
        'use_semantic': True,
        'use_final_refine': False,
        'semantic_feature_dim': 64,
        'semantic_pretrained': False,
        'num_classes': 19,
        'semantic_downsample': 1,
        'transformer_embed_dim': 96,
        'transformer_out_dim': 768,
        'num_curve_points': 8,
    }

    try:
        optimized_model = OptimizedEnhancementModel(optimized_cfg).to(device)

        with torch.no_grad():
            enhanced, components = optimized_model(dummy, return_components=True)

        print(f"  Enhanced output shape: {enhanced.shape}")
        print(f"  Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")

        if components:
            print(f"\n  Components returned:")
            for key, val in components.items():
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: {val.shape}")
                elif isinstance(val, dict):
                    print(f"    {key}: dict with {len(val)} items")

        # Count parameters
        total_params = sum(p.numel() for p in optimized_model.parameters())
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

        print("\n  All tests passed for OptimizedEnhancementModel!")

    except Exception as e:
        print(f"  Warning: OptimizedEnhancementModel test failed: {e}")
        print("  This is expected if retinex_semantic.py or curve_transformer.py are not available")

    print("\n" + "="*70)
    print("All component tests completed!")
    print("="*70)
