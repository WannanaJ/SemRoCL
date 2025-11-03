"""
Enhanced Model Components for SemRoCL Stage 2
Integrates Multi-scale Enhancement and Semantic Guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class EnhancementGenerator(nn.Module):
    """
    Main generator for low-light image enhancement
    Supports both basic and multi-scale modes
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration
        if config is None:
            config = {
                'enhancer_channels': 32,
                'num_iterations': 8,
                'semantic_channels': 64,
                'use_semantic': True,
                'use_multiscale': False,  # New: multi-scale option
                'num_scales': 3
            }
        
        self.n_channels = config.get('enhancer_channels', 32)
        self.num_iterations = config.get('num_iterations', 8)
        self.use_semantic = config.get('use_semantic', True)
        self.semantic_channels = config.get('semantic_channels', 64)
        self.use_multiscale = config.get('use_multiscale', False)
        self.num_scales = config.get('num_scales', 3)
        
        if self.use_multiscale:
            # Multi-scale architecture
            self._build_multiscale_enhancer()
        else:
            # Single-scale architecture (original)
            self._build_single_scale_enhancer()
    
    def _build_single_scale_enhancer(self):
        """Build single-scale enhancer (original architecture)"""
        # Feature extraction network
        self.conv1 = nn.Conv2d(3, self.n_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        
        # Semantic fusion
        if self.use_semantic:
            self.semantic_fusion = nn.Sequential(
                nn.Conv2d(self.n_channels, self.n_channels, 1, 1, 0),  # 32 -> 32
                nn.ReLU(inplace=True)
            )
        
        # Curve parameter estimation
        self.conv5 = nn.Conv2d(self.n_channels, 3 * self.num_iterations, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def _build_multiscale_enhancer(self):
        """Build multi-scale enhancer"""
        # Create enhancers for each scale
        self.scale_enhancers = nn.ModuleList()
        
        for i in range(self.num_scales):
            # Each scale has its own feature extraction
            scale_net = nn.ModuleDict({
                'conv1': nn.Conv2d(3, self.n_channels, 3, 1, 1, bias=True),
                'conv2': nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True),
                'conv3': nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True),
                'conv4': nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True),
            })
            
            # Semantic fusion for each scale
            if self.use_semantic:
                scale_net['semantic_fusion'] = nn.Sequential(
                    # 动态计算输入通道数：generator特征(32) + 语义特征(64) = 96
                    nn.Conv2d(self.n_channels + 64, self.n_channels, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            
            # Curve estimation for each scale
            scale_net['conv5'] = nn.Conv2d(self.n_channels, 3 * self.num_iterations, 3, 1, 1, bias=True)
            
            self.scale_enhancers.append(scale_net)
        
        # Multi-scale fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * self.num_scales, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def _enhance_single_scale(self, x, semantic_features=None, confidence_map=None):
        """Single-scale enhancement"""
        # Feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Semantic feature fusion
        if self.use_semantic and semantic_features is not None:
            # Resize semantic features to match
            semantic_features = F.interpolate(
                semantic_features,
                size=x4.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Concatenate and fuse
            # ensure size match
            if semantic_features.shape[2:] != x4.shape[2:]:
               semantic_features = F.interpolate(semantic_features, size=x4.shape[2:], mode="bilinear", align_corners=False)

            # add (semantic modulation)
            x4 = x4 + semantic_features   #  NOT concat
            x4 = self.semantic_fusion(x4)

            
            # Apply confidence weighting
            if confidence_map is not None:
                confidence_map = F.interpolate(
                    confidence_map,
                    size=x4.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                x4 = x4 * confidence_map
        
        # Estimate curve parameters
        curves = self.tanh(self.conv5(x4))
        
        # Apply iterative curve adjustment
        enhanced = x
        for i in range(self.num_iterations):
            curve_params = curves[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + curve_params * enhanced * (1 - enhanced)
            enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced
    
    def _enhance_multiscale(self, x, semantic_features=None, confidence_map=None):
        """Multi-scale enhancement"""
        B, C, H, W = x.shape
        enhanced_scales = []
        
        for scale_idx, scale_net in enumerate(self.scale_enhancers):
            # Compute scale factor
            scale = 2 ** scale_idx
            
            # Downsample input
            x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            
            # Process with scale-specific network
            x1 = self.relu(scale_net['conv1'](x_scaled))
            x2 = self.relu(scale_net['conv2'](x1))
            x3 = self.relu(scale_net['conv3'](x2))
            x4 = self.relu(scale_net['conv4'](x3))
            
            # Semantic fusion
            if self.use_semantic and semantic_features is not None:
                semantic_scaled = F.interpolate(
                    semantic_features,
                    size=x4.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                # 确保语义特征尺寸匹配
                if semantic_scaled.shape[2:] != x4.shape[2:]:
                    semantic_scaled = F.interpolate(semantic_scaled, size=x4.shape[2:], mode='bilinear', align_corners=False)

                # 拼接特征（32 通道 + 64 通道 = 96 通道）
                x4 = torch.cat([x4, semantic_scaled], dim=1)
                x4 = scale_net['semantic_fusion'](x4)
                
                if confidence_map is not None:
                    conf_scaled = F.interpolate(
                        confidence_map,
                        size=x4.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    x4 = x4 * conf_scaled
            
            # Estimate curves
            curves = self.tanh(scale_net['conv5'](x4))
            
            # Apply curves
            enhanced_scaled = x_scaled
            for i in range(self.num_iterations):
                curve_params = curves[:, i*3:(i+1)*3, :, :]
                enhanced_scaled = enhanced_scaled + curve_params * enhanced_scaled * (1 - enhanced_scaled)
                enhanced_scaled = torch.clamp(enhanced_scaled, 0, 1)
            
            # Upsample to original size
            enhanced_upsampled = F.interpolate(
                enhanced_scaled,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            enhanced_scales.append(enhanced_upsampled)
        
        # Fuse multi-scale results
        multi_scale_features = torch.cat(enhanced_scales, dim=1)
        enhanced_final = self.fusion(multi_scale_features)
        
        return enhanced_final
    
    def forward(self, x, semantic_features=None, confidence_map=None):
        """
        Forward pass
        
        Args:
            x: Input low-light image (B, 3, H, W)
            semantic_features: Optional semantic guidance (B, C, H, W)
            confidence_map: Optional confidence map (B, 1, H, W)
        
        Returns:
            Enhanced image (B, 3, H, W)
        """
        if self.use_multiscale:
            return self._enhance_multiscale(x, semantic_features, confidence_map)
        else:
            return self._enhance_single_scale(x, semantic_features, confidence_map)


class SemanticGuidanceModule(nn.Module):
    """
    Semantic guidance module with full segmentation head
    Integrates SegFormer for semantic understanding
    """
    
    def __init__(self, num_classes=19, pretrained=True, output_features=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_features = output_features
        
        # Load SegFormer-B0
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        # 特征变换层 - 固定的投影链（160通道输入，适配 SegFormer-B0 的 hidden_states[2]）
        self.feature_projector = nn.Sequential(
            nn.Conv2d(160, 128, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(True)
        )
        
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.segformer = SegformerForSemanticSegmentation(config)
        
        # Uncertainty/confidence estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            seg_logits: Segmentation logits (B, num_classes, H, W)
            confidence: Confidence map (B, 1, H, W)
            features: Semantic features (B, 64, H, W) for enhancement guidance
        """
        B, C, H, W = x.shape
        
        # Get segmentation and features
        outputs = self.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        seg_logits = outputs.logits
        
        # Upsample to original size
        seg_logits = F.interpolate(
            seg_logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # Compute confidence
        confidence = self.uncertainty_head(seg_logits)
        
        # Extract features for enhancement guidance
        if self.output_features:
            # 使用主干最后一层特征（160通道）
            hidden_states = outputs.hidden_states
            features = hidden_states[2]  # SegFormer-B0 主干最后一层，160通道
            
            # 先上采样到目标尺寸
            features = F.interpolate(
                features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            # 通过投影链将特征转换到64通道
            features = self.feature_projector(features)
            
            # 添加调试信息
            print(f"Features shape after projection: {features.shape}")
        else:
            features = None
        
        return seg_logits, confidence, features


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3, use_spectral_norm=True):
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator
        layers = []
        
        # First layer
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
            ))
        else:
            layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)
                ))
            else:
                layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(ndf * nf_mult))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1)
            ))
        else:
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("Testing Enhanced Models")
    print("="*60)
    
    # Test 1: Single-scale generator (original)
    print("\n1. Testing Single-scale Generator:")
    config_single = {
        'enhancer_channels': 32,
        'num_iterations': 8,
        'semantic_channels': 64,
        'use_semantic': True,
        'use_multiscale': False
    }
    
    generator_single = EnhancementGenerator(config_single).to(device)
    x = torch.randn(2, 3, 384, 384).to(device)
    semantic_features = torch.randn(2, 64, 384, 384).to(device)
    
    enhanced_single = generator_single(x, semantic_features)
    print(f"   Input: {x.shape}")
    print(f"   Output: {enhanced_single.shape}")
    print(f"   ✓ Single-scale works!")
    
    # Test 2: Multi-scale generator
    print("\n2. Testing Multi-scale Generator:")
    config_multi = {
        'enhancer_channels': 32,
        'num_iterations': 8,
        'semantic_channels': 64,
        'use_semantic': True,
        'use_multiscale': True,
        'num_scales': 3
    }
    
    generator_multi = EnhancementGenerator(config_multi).to(device)
    enhanced_multi = generator_multi(x, semantic_features)
    print(f"   Input: {x.shape}")
    print(f"   Output: {enhanced_multi.shape}")
    print(f"   ✓ Multi-scale works!")
    
    # Test 3: Semantic guidance module
    print("\n3. Testing Semantic Guidance Module:")
    semantic_module = SemanticGuidanceModule(num_classes=19, pretrained=False).to(device)
    seg_logits, confidence, features = semantic_module(x)
    print(f"   Input: {x.shape}")
    print(f"   Seg logits: {seg_logits.shape}")
    print(f"   Confidence: {confidence.shape}")
    print(f"   Features: {features.shape}")
    print(f"   ✓ Semantic guidance works!")
    
    # Test 4: Discriminator
    print("\n4. Testing Discriminator:")
    discriminator = Discriminator().to(device)
    disc_pred = discriminator(enhanced_single)
    print(f"   Input: {enhanced_single.shape}")
    print(f"   Output: {disc_pred.shape}")
    print(f"   ✓ Discriminator works!")
    
    # Count parameters
    print("\n" + "="*60)
    print("Parameter Statistics:")
    print("="*60)
    single_params = sum(p.numel() for p in generator_single.parameters())
    multi_params = sum(p.numel() for p in generator_multi.parameters())
    semantic_params = sum(p.numel() for p in semantic_module.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Single-scale Generator: {single_params:,}")
    print(f"Multi-scale Generator: {multi_params:,}")
    print(f"Semantic Module: {semantic_params:,}")
    print(f"Discriminator: {disc_params:,}")
    print(f"Total (Multi-scale setup): {multi_params + semantic_params + disc_params:,}")
    
    print("\n✓ All tests passed!")