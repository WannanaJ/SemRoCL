"""
PatchGAN Discriminator for Adversarial Training
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for local realism assessment
    Outputs a matrix of predictions rather than a single value
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            ndf: Number of filters in first conv layer
            n_layers: Number of discriminator layers
        """
        super(PatchGANDiscriminator, self).__init__()
        
        # Build discriminator network
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
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


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better gradient flow
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3, num_D=3):
        """
        Args:
            in_channels: Number of input channels
            ndf: Number of base filters
            n_layers: Layers per discriminator
            num_D: Number of discriminators at different scales
        """
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_D = num_D
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            netD = PatchGANDiscriminator(in_channels, ndf, n_layers)
            self.discriminators.append(netD)
        
        # Downsampling for coarser scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        """
        Forward pass through all scales
        
        Args:
            x: Input image
        
        Returns:
            List of predictions at different scales
        """
        results = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            
            pred = discriminator(x)
            results.append(pred)
        
        return results


class SpectralNormDiscriminator(nn.Module):
    """
    Discriminator with spectral normalization for training stability
    """
    
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(SpectralNormDiscriminator, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Test discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic PatchGAN
    disc = PatchGANDiscriminator(in_channels=3, ndf=64, n_layers=3).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    pred = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"PatchGAN prediction shape: {pred.shape}")
    
    # Test multi-scale discriminator
    ms_disc = MultiScaleDiscriminator(in_channels=3, ndf=64, n_layers=3, num_D=3).to(device)
    preds = ms_disc(x)
    print(f"\nMulti-scale discriminator:")
    for i, pred in enumerate(preds):
        print(f"Scale {i} prediction shape: {pred.shape}")
    
    # Test spectral norm discriminator
    sn_disc = SpectralNormDiscriminator(in_channels=3, ndf=64, n_layers=3).to(device)
    pred = sn_disc(x)
    print(f"\nSpectral norm discriminator prediction shape: {pred.shape}")
