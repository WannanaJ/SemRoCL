"""
Retinex-Semantic Fusion Module (P0 Optimization)
=================================================

Physical-guided semantic enhancement for low-light images.
Combines Retinex theory (I = R * L) with deep semantic guidance.

Key Innovations:
1. Learnable Retinex decomposition with semantic boundary preservation
2. Semantic-guided reflectance enhancement
3. Adaptive illumination correction based on semantic context

Target Metrics for TIP Submission:
- PSNR: +0.8~1.2 dB improvement
- Structure preservation with semantic consistency
- Interpretable decomposition for ablation studies

Author: SemRoCL Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexDecomposition(nn.Module):
    """
    Learnable Retinex Decomposition Network
    Decomposes low-light image into Reflectance and Illumination: I = R * L

    Based on DecomNet with improvements for semantic integration
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # Reflectance estimation network
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.Sigmoid()  # Reflectance range [0, 1]
        )

        # Illumination estimation network
        self.illumination_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),  # Single-channel illumination map
            nn.Sigmoid()  # Illumination range [0, 1]
        )

    def forward(self, img):
        """
        Decompose image into reflectance and illumination

        Args:
            img: Low-light image (B, 3, H, W), range [0, 1]

        Returns:
            R: Reflectance map (B, 3, H, W)
            L: Illumination map (B, 1, H, W)
        """
        R = self.reflectance_net(img)
        L = self.illumination_net(img)

        return R, L


class SemanticGuidedRefine(nn.Module):
    """
    Semantic-guided reflectance refinement
    Preserves semantic boundaries while enhancing details
    """
    def __init__(self, semantic_dim=512, out_channels=3):
        super().__init__()

        # Semantic feature projection
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_dim, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Reflectance refinement with semantic guidance
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh()  # Residual output range [-1, 1]
        )

    def forward(self, R, semantic_feat):
        """
        Refine reflectance using semantic guidance

        Args:
            R: Reflectance map (B, 3, H, W)
            semantic_feat: Semantic features (B, C, H', W')

        Returns:
            R_enhanced: Enhanced reflectance (B, 3, H, W)
        """
        # Spatial alignment
        if semantic_feat.shape[2:] != R.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=R.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Project semantic features
        sem_proj = self.semantic_proj(semantic_feat)

        # Concatenate and refine
        fused = torch.cat([R, sem_proj], dim=1)
        residual = self.refine(fused)

        # Residual connection with scaling
        R_enhanced = torch.clamp(R + 0.2 * residual, 0, 1)

        return R_enhanced


class SemanticGuidedIllumination(nn.Module):
    """
    Semantic-guided illumination adjustment
    Adapts enhancement strength based on semantic context
    """
    def __init__(self, semantic_dim=512):
        super().__init__()

        # Adaptive illumination adjustment
        self.illumination_adjust = nn.Sequential(
            nn.Conv2d(1 + semantic_dim, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()  # Adjusted illumination range [0, 1]
        )

    def forward(self, L, semantic_feat):
        """
        Adjust illumination using semantic guidance

        Args:
            L: Illumination map (B, 1, H, W)
            semantic_feat: Semantic features (B, C, H', W')

        Returns:
            L_corrected: Corrected illumination (B, 1, H, W)
        """
        # Spatial alignment
        if semantic_feat.shape[2:] != L.shape[2:]:
            semantic_feat = F.interpolate(
                semantic_feat, size=L.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Concatenate and adjust
        fused = torch.cat([L, semantic_feat], dim=1)
        L_corrected = self.illumination_adjust(fused)

        return L_corrected


class RetinexSemanticFusion(nn.Module):
    """
    Complete Retinex-Semantic Fusion Module

    Pipeline:
    1. Decompose input into R and L (I = R * L)
    2. Enhance R using semantic guidance (preserve boundaries)
    3. Adjust L using semantic context (adaptive correction)
    4. Reconstruct enhanced image (I_enh = R_enh * L_cor)

    Features:
    - Physically interpretable decomposition
    - Semantic boundary preservation
    - Adaptive enhancement per region
    - Compatible with physical constraint losses
    """
    def __init__(self, semantic_dim=512):
        super().__init__()

        self.decomposition = RetinexDecomposition()
        self.reflectance_enhance = SemanticGuidedRefine(semantic_dim)
        self.illumination_adjust = SemanticGuidedIllumination(semantic_dim)

    def forward(self, low_img, semantic_feat):
        """
        Perform Retinex-semantic fusion enhancement

        Args:
            low_img: Low-light image (B, 3, H, W), range [0, 1]
            semantic_feat: Semantic features (B, C, H', W')

        Returns:
            enhanced: Enhanced image (B, 3, H, W)
            components: Dictionary containing:
                - R: Original reflectance
                - L: Original illumination
                - R_enhanced: Enhanced reflectance
                - L_corrected: Corrected illumination
                - reconstruction: R * L (for debugging)
        """
        # 1. Retinex decomposition
        R, L = self.decomposition(low_img)

        # 2. Semantic-guided reflectance enhancement
        R_enhanced = self.reflectance_enhance(R, semantic_feat)

        # 3. Semantic-guided illumination adjustment
        L_corrected = self.illumination_adjust(L, semantic_feat)

        # 4. Reconstruction
        enhanced = R_enhanced * L_corrected.expand_as(R_enhanced)

        # Clamp to valid range
        enhanced = torch.clamp(enhanced, 0, 1)

        # Package components for loss computation and visualization
        components = {
            'R': R,
            'L': L,
            'R_enhanced': R_enhanced,
            'L_corrected': L_corrected,
            'reconstruction': R * L.expand_as(R)
        }

        return enhanced, components


# ==============================================================================
# Testing and Validation
# ==============================================================================

def test_retinex_semantic():
    """Unit test for Retinex-Semantic Fusion"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create module
    semantic_dim = 512
    module = RetinexSemanticFusion(semantic_dim).to(device)

    # Test inputs
    batch_size = 2
    height, width = 256, 256
    low_img = torch.rand(batch_size, 3, height, width).to(device)
    semantic_feat = torch.rand(batch_size, semantic_dim, height // 4, width // 4).to(device)

    # Forward pass
    print("Testing Retinex-Semantic Fusion...")
    with torch.no_grad():
        enhanced, components = module(low_img, semantic_feat)

    # Validate outputs
    assert enhanced.shape == (batch_size, 3, height, width), "Enhanced shape mismatch"
    assert components['R'].shape == (batch_size, 3, height, width), "R shape mismatch"
    assert components['L'].shape == (batch_size, 1, height, width), "L shape mismatch"
    assert components['R_enhanced'].shape == (batch_size, 3, height, width), "R_enhanced shape mismatch"
    assert components['L_corrected'].shape == (batch_size, 1, height, width), "L_corrected shape mismatch"

    # Check value ranges
    assert torch.all((enhanced >= 0) & (enhanced <= 1)), "Enhanced values out of range"
    assert torch.all((components['R'] >= 0) & (components['R'] <= 1)), "R values out of range"
    assert torch.all((components['L'] >= 0) & (components['L'] <= 1)), "L values out of range"

    print("[OK] All tests passed!")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"  R range: [{components['R'].min():.3f}, {components['R'].max():.3f}]")
    print(f"  L range: [{components['L'].min():.3f}, {components['L'].max():.3f}]")

    return module, enhanced, components


if __name__ == '__main__':
    # Run tests
    test_retinex_semantic()
