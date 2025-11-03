"""
Model Components for SemRoCL Stage 2
Integrates Encoder, Enhancer, Semantic Head, and Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancementGenerator(nn.Module):
    """
    Main generator for low-light image enhancement
    Combines encoder features, semantic guidance, and curve-based enhancement
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration
        if config is None:
            config = {
                'enhancer_channels': 32,
                'num_iterations': 8,
                'semantic_channels': 64,
                'use_semantic': True
            }
        
        self.n_channels = config.get('enhancer_channels', 32)
        self.num_iterations = config.get('num_iterations', 8)
        self.use_semantic = config.get('use_semantic', True)
        self.semantic_channels = config.get('semantic_channels', 64)
        
        # Feature extraction network (from CurveEnhancer)
        self.conv1 = nn.Conv2d(3, self.n_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(self.n_channels, self.n_channels, 3, 1, 1, bias=True)
        
        # Semantic fusion (if enabled)
        if self.use_semantic:
            self.semantic_fusion = nn.Sequential(
                nn.Conv2d(self.n_channels + self.semantic_channels, self.n_channels, 1, 1, 0),
                nn.ReLU(inplace=True)
            )
        
        # Curve parameter estimation
        self.conv5 = nn.Conv2d(self.n_channels, 3 * self.num_iterations, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
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
        # Feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        
        # Semantic fusion only if features are provided and enabled
        if self.use_semantic and semantic_features is not None:
            # Resize semantic features to match spatial size if needed
            if semantic_features.shape[2:] != x4.shape[2:]:
                semantic_features = F.interpolate(
                    semantic_features,
                    size=x4.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            # Fuse only when channel size matches expected semantic_channels
            if semantic_features.shape[1] == self.semantic_channels:
                x4 = torch.cat([x4, semantic_features], dim=1)
                x4 = self.semantic_fusion(x4)
            # If channels don't match, skip fusion gracefully
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
        
        return enhanced


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3, use_spectral_norm=True):
        """
        Args:
            in_channels: Number of input channels
            ndf: Number of base filters
            n_layers: Number of discriminator layers
            use_spectral_norm: Use spectral normalization for stability
        """
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator
        layers = []
        
        # First layer (no normalization)
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
        
        # Output layer (1 channel prediction map)
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Patch predictions (B, 1, H', W')
        """
        return self.model(x)


if __name__ == '__main__':
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test generator
    config = {
        'enhancer_channels': 32,
        'num_iterations': 8,
        'semantic_channels': 64,
        'use_semantic': True
    }
    
    generator = EnhancementGenerator(config).to(device)
    discriminator = Discriminator().to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384).to(device)
    semantic_features = torch.randn(2, 64, 384, 384).to(device)
    
    # Generator
    enhanced = generator(x, semantic_features)
    print(f"Input shape: {x.shape}")
    print(f"Enhanced shape: {enhanced.shape}")
    
    # Discriminator
    disc_pred = discriminator(enhanced)
    print(f"Discriminator prediction shape: {disc_pred.shape}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
