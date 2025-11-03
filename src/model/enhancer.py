"""
Pixel-level Curve Estimation Enhancer
Inspired by Zero-DCE with semantic guidance integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurveEnhancer(nn.Module):
    """
    Curve-based image enhancement with semantic guidance
    
    Estimates pixel-wise curves for iterative enhancement
    Zero-DCE style but with semantic feature fusion
    """
    
    def __init__(self, n_channels=32, num_iterations=8, use_semantic=True, semantic_channels=64):
        """
        Args:
            n_channels: Number of base channels
            num_iterations: Number of curve iterations
            use_semantic: Whether to use semantic guidance
            semantic_channels: Number of channels in semantic features (default 64 for SegFormer-B0)
        """
        super(CurveEnhancer, self).__init__()
        
        self.num_iterations = num_iterations
        self.use_semantic = use_semantic
        
        # Feature extraction network
        self.conv1 = nn.Conv2d(3, n_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        
        # Semantic fusion (if enabled)
        if self.use_semantic:
            self.semantic_fusion = nn.Sequential(
                nn.Conv2d(n_channels + semantic_channels, n_channels, 1, 1, 0),  # Fixed: use semantic_channels parameter
                nn.ReLU(inplace=True)
            )
        
        # Curve parameter estimation
        # Outputs 3 * num_iterations channel map (RGB curves for each iteration)
        self.conv5 = nn.Conv2d(n_channels, 3 * num_iterations, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x, semantic_features=None, confidence_map=None):
        """
        Forward pass
        
        Args:
            x: Input low-light image (B, 3, H, W)
            semantic_features: Semantic guidance features (B, C, H, W)
            confidence_map: Confidence/uncertainty map (B, 1, H, W)
        
        Returns:
            enhanced: Enhanced image (B, 3, H, W)
            curves: Estimated curve parameters (B, 3*num_iterations, H, W)
        """
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
            x4 = torch.cat([x4, semantic_features], dim=1)
            x4 = self.semantic_fusion(x4)
            
            # Apply confidence weighting if available
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
            # Get curve parameters for this iteration
            curve_params = curves[:, i*3:(i+1)*3, :, :]
            
            # Apply curve: I_out = I_in + curve_params * I_in * (1 - I_in)
            enhanced = enhanced + curve_params * enhanced * (1 - enhanced)
            
            # Clamp to valid range
            enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced, curves
    
    def enhance_with_encoder_features(self, x, encoder_features, semantic_features=None, confidence_map=None):
        """
        Enhanced version using both encoder and semantic features
        
        Args:
            x: Input image
            encoder_features: Features from pretrained encoder (B, feat_dim)
            semantic_features: Semantic segmentation features
            confidence_map: Uncertainty map
        
        Returns:
            Enhanced image and curves
        """
        # This would be extended to use encoder features for additional guidance
        # For now, use standard enhancement with semantic guidance
        return self.forward(x, semantic_features, confidence_map)


class MultiScaleEnhancer(nn.Module):
    """
    Multi-scale curve enhancer with pyramid processing
    """
    
    def __init__(self, n_channels=32, num_iterations=8, num_scales=3, semantic_channels=64):
        super(MultiScaleEnhancer, self).__init__()
        
        self.num_scales = num_scales
        
        # Create enhancers for each scale
        self.enhancers = nn.ModuleList([
            CurveEnhancer(n_channels, num_iterations, use_semantic=True, semantic_channels=semantic_channels)
            for _ in range(num_scales)
        ])
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_scales, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, semantic_features=None, confidence_map=None):
        """
        Multi-scale enhancement
        
        Args:
            x: Input image
            semantic_features: Semantic features
            confidence_map: Confidence map
        
        Returns:
            Final enhanced image
        """
        B, C, H, W = x.shape
        
        enhanced_scales = []
        
        for i, enhancer in enumerate(self.enhancers):
            # Compute scale factor
            scale = 2 ** i
            
            # Downsample input
            x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
            
            # Downsample semantic features if provided
            if semantic_features is not None:
                semantic_scaled = F.interpolate(
                    semantic_features, 
                    scale_factor=1/scale, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                semantic_scaled = None
            
            # Enhance at this scale
            enhanced_scaled, _ = enhancer(x_scaled, semantic_scaled, confidence_map)
            
            # Upsample back to original size
            enhanced_upsampled = F.interpolate(
                enhanced_scaled, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            
            enhanced_scales.append(enhanced_upsampled)
        
        # Concatenate all scales
        multi_scale_features = torch.cat(enhanced_scales, dim=1)
        
        # Fuse multi-scale results
        enhanced_final = self.fusion(multi_scale_features)
        
        return enhanced_final


class AdaptiveEnhancer(nn.Module):
    """
    Adaptive enhancer with learned attention
    """
    
    def __init__(self, n_channels=32, num_iterations=8, semantic_channels=64):
        super(AdaptiveEnhancer, self).__init__()
        
        self.base_enhancer = CurveEnhancer(n_channels, num_iterations, use_semantic=True, semantic_channels=semantic_channels)
        
        # Attention module for adaptive enhancement
        self.attention = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, semantic_features=None, confidence_map=None):
        """
        Adaptive enhancement based on input content
        """
        # Compute attention map
        attention_map = self.attention(x)
        
        # Base enhancement
        enhanced, curves = self.base_enhancer(x, semantic_features, confidence_map)
        
        # Adaptive blending
        output = attention_map * enhanced + (1 - attention_map) * x
        
        return output, curves


if __name__ == '__main__':
    # Test enhancer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic enhancer with correct semantic channel size
    enhancer = CurveEnhancer(n_channels=32, num_iterations=8, use_semantic=True, semantic_channels=64).to(device)
    
    x = torch.randn(2, 3, 512, 512).to(device)
    semantic_features = torch.randn(2, 64, 512, 512).to(device)  # Changed from 128 to 64
    confidence_map = torch.randn(2, 1, 512, 512).to(device)
    
    enhanced, curves = enhancer(x, semantic_features, confidence_map)
    
    print(f"Input shape: {x.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Curves shape: {curves.shape}")
    
    # Test multi-scale enhancer
    ms_enhancer = MultiScaleEnhancer(n_channels=32, num_iterations=8, num_scales=3, semantic_channels=64).to(device)
    enhanced_ms = ms_enhancer(x, semantic_features, confidence_map)
    print(f"Multi-scale enhanced shape: {enhanced_ms.shape}")