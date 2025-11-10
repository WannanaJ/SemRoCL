"""
Optimized Loss Functions for Stage 2 Training
Fixed NaN handling and integrated with actual model architecture
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ColorLoss(nn.Module):
    """
    Color consistency loss with improved stability
    """
    def __init__(self, weight_exp=10, weight_color=5, weight_tv=200):
        super().__init__()
        self.weight_exp = weight_exp
        self.weight_color = weight_color
        self.weight_tv = weight_tv
        self.eps = 1e-6
    
    def exposure_loss(self, x):
        """Exposure control loss"""
        mean_val = torch.mean(x, dim=[2, 3], keepdim=True)
        target = 0.62
        loss = F.smooth_l1_loss(mean_val, torch.ones_like(mean_val) * target)
        return torch.clamp(loss, 0, 1.0)
    
    def color_constancy_loss(self, x):
        """Color constancy loss"""
        mean_rgb = torch.mean(x, dim=[2, 3], keepdim=True)
        mr = mean_rgb[:, 0:1]
        mg = mean_rgb[:, 1:2]
        mb = mean_rgb[:, 2:3]
        
        d_rg = torch.clamp((mr - mg) ** 2, 0, 1.0)
        d_rb = torch.clamp((mr - mb) ** 2, 0, 1.0)
        d_gb = torch.clamp((mg - mb) ** 2, 0, 1.0)
        
        loss = torch.sqrt(d_rg + d_rb + d_gb + self.eps)
        return torch.mean(loss)
    
    def total_variation_loss(self, x):
        """Total variation loss for smoothness"""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        
        loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))
        return torch.clamp(loss, 0, 0.5)
    
    def forward(self, enhanced, original=None):
        """
        Compute color loss
        
        Args:
            enhanced: Enhanced image (B, 3, H, W)
            original: Original image (optional)
        
        Returns:
            Total color loss
        """
        # Clamp input to valid range
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # Compute individual losses
        exp_loss = self.exposure_loss(enhanced) * self.weight_exp
        color_loss = self.color_constancy_loss(enhanced) * self.weight_color
        tv_loss = self.total_variation_loss(enhanced) * self.weight_tv
        
        total_loss = exp_loss + color_loss + tv_loss
        total_loss = torch.clamp(total_loss, 0, 10.0)
        
        return total_loss


class SemanticConsistencyLoss(nn.Module):
    """
    Semantic feature consistency loss
    Uses features from semantic segmentation head
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        self.max_loss = 2.0
    
    def forward(self, feat_low, feat_high):
        """
        Compute semantic consistency loss
        
        Args:
            feat_low: Features from low-light image
            feat_high: Features from high-quality image
        
        Returns:
            Consistency loss
        """
        # Normalize features
        feat_low_norm = F.normalize(feat_low, dim=1, p=2, eps=self.eps)
        feat_high_norm = F.normalize(feat_high, dim=1, p=2, eps=self.eps)
        
        # Cosine similarity
        similarity = torch.sum(feat_low_norm * feat_high_norm, dim=1)
        similarity = torch.clamp(similarity, -1.0 + self.eps, 1.0 - self.eps)
        
        # Loss is 1 - similarity (want high similarity)
        loss = 1.0 - similarity.mean()
        loss = torch.clamp(loss, 0, self.max_loss)
        
        return loss


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss using FFT
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    
    def forward(self, pred, target):
        """
        Compute frequency loss
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            Frequency domain loss
        """
        # Clamp inputs
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Convert to grayscale
        pred_gray = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        try:
            # FFT
            pred_fft = torch.fft.fft2(pred_gray)
            target_fft = torch.fft.fft2(target_gray)
            
            # Magnitude with larger epsilon for stability
            pred_mag = torch.clamp(torch.abs(pred_fft), min=1e-5)
            target_mag = torch.clamp(torch.abs(target_fft), min=1e-5)
            
            # Log magnitude loss (more stable)
            loss = F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))
            loss = torch.clamp(loss, 0, 1.0)
            
            return loss
            
        except Exception:
            # Return small constant if FFT fails
            return torch.tensor(0.01, device=pred.device, requires_grad=True)


class NoiseSuppressionLoss(nn.Module):
    """
    Penalize excessive high-frequency magnitude relative to target (reduces halos/noise).
    """
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=torch.float32
        )
        self.register_buffer('lap_kernel', kernel.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        c = pred.size(1)
        kernel = self.lap_kernel.to(pred.device, dtype=pred.dtype).repeat(c, 1, 1, 1)

        pred_hp = F.conv2d(pred, kernel, padding=1, groups=c)
        target_hp = F.conv2d(target, kernel, padding=1, groups=c)

        excess = torch.relu(torch.abs(pred_hp) - torch.abs(target_hp))
        return torch.clamp(excess.mean(), 0.0, 1.0)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG16_Weights
        
        # Use new weights API
        vgg_features = cast(nn.Sequential, models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*[vgg_features[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg_features[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg_features[i] for i in range(9, 16)])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eps = 1e-8
        
        # ImageNet normalization
        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image
            target: Target image
        
        Returns:
            Perceptual loss
        """
        # Clamp inputs
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Normalize
        pred_norm = (pred - self.mean) / (self.std + self.eps)
        target_norm = (target - self.mean) / (self.std + self.eps)
        
        try:
            # Extract features
            pred_f1 = self.slice1(pred_norm)
            pred_f2 = self.slice2(pred_f1)
            pred_f3 = self.slice3(pred_f2)
            
            target_f1 = self.slice1(target_norm)
            target_f2 = self.slice2(target_f1)
            target_f3 = self.slice3(target_f2)
            
            # Compute losses
            loss1 = F.l1_loss(pred_f1, target_f1)
            loss2 = F.l1_loss(pred_f2, target_f2)
            loss3 = F.l1_loss(pred_f3, target_f3)
            
            total_loss = loss1 + loss2 + loss3
            total_loss = torch.clamp(total_loss, 0, 5.0)
            
            return total_loss
            
        except Exception:
            return torch.tensor(0.1, device=pred.device, requires_grad=True)


class TaskAlignmentLoss(nn.Module):
    """
    Task alignment loss for semantic consistency
    """
    def __init__(self, num_classes=19, temperature=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = 1e-8
    
    def forward(self, pred_seg, target_seg):
        """
        Compute task alignment loss using KL divergence
        
        Args:
            pred_seg: Predicted segmentation logits
            target_seg: Target segmentation logits
        
        Returns:
            KL divergence loss
        """
        # Softmax with temperature
        pred_prob = F.softmax(pred_seg / self.temperature, dim=1)
        target_prob = F.softmax(target_seg / self.temperature, dim=1)
        
        # KL divergence
        log_pred = torch.log(pred_prob + self.eps)
        kl_loss = F.kl_div(log_pred, target_prob, reduction='batchmean')
        
        kl_loss = torch.clamp(kl_loss, 0, 2.0)
        
        return kl_loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma):
        """Create Gaussian kernel"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        Compute SSIM loss
        
        Args:
            img1: First image
            img2: Second image
        
        Returns:
            SSIM loss (1 - SSIM)
        """
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute means
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            ssim_value = ssim_map.mean()
        else:
            ssim_value = ssim_map.mean(1).mean(1).mean(1)
        
        # Return loss (1 - SSIM)
        return 1 - ssim_value


# ==============================================================================
# P0 Optimization: Advanced Perceptual Loss (LPIPS + DISTS + Style)
# ==============================================================================

class AdvancedPerceptualLoss(nn.Module):
    """
    Multi-scale perceptual loss combining:
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - DISTS (Deep Image Structure and Texture Similarity)
    - VGG Style Loss (Gram matrix based)

    Target: Achieve LPIPS < 0.05 for TIP submission
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # 1. LPIPS - Perceptual similarity
        try:
            import lpips
            self.lpips_net = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
            self.use_lpips = True
        except ImportError:
            print("Warning: lpips not installed. Install with: pip install lpips")
            self.use_lpips = False

        # 2. DISTS - Structure and texture similarity
        try:
            import DISTS_pytorch as DISTS_module
            self.dists_net = DISTS_module.DISTS().to(device).eval()
            self.use_dists = True
        except ImportError:
            print("Warning: DISTS_pytorch not installed. Install with: pip install DISTS-pytorch")
            self.use_dists = False

        # 3. VGG for style loss
        from torchvision import models
        from torchvision.models import VGG19_Weights
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Extract multi-scale features for style loss
        self.vgg_layers = nn.ModuleDict({
            'relu1_2': nn.Sequential(*[vgg[i] for i in range(4)]),
            'relu2_2': nn.Sequential(*[vgg[i] for i in range(9)]),
            'relu3_4': nn.Sequential(*[vgg[i] for i in range(18)]),
            'relu4_4': nn.Sequential(*[vgg[i] for i in range(27)])
        }).to(device).eval()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Loss weights (optimized for TIP metrics)
        self.w_lpips = 0.40
        self.w_dists = 0.30
        self.w_style = 0.30

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def gram_matrix(self, features):
        """Compute Gram matrix for style loss"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def compute_style_loss(self, pred, target):
        """Multi-scale style loss using Gram matrices"""
        # Normalize inputs
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        loss = 0.0
        for layer_name, layer in self.vgg_layers.items():
            pred_feat = layer(pred_norm)
            target_feat = layer(target_norm)

            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)

            loss += F.mse_loss(pred_gram, target_gram)

        return loss / len(self.vgg_layers)

    def forward(self, pred, target):
        """
        Compute advanced perceptual loss

        Args:
            pred: Enhanced image (B, 3, H, W), range [0, 1] or [-1, 1]
            target: Target image (B, 3, H, W), range [0, 1] or [-1, 1]

        Returns:
            total_loss: Weighted combination of perceptual losses
            components: Dictionary with individual loss values
        """
        # Ensure [0, 1] range
        pred = torch.clamp(pred, 0, 1) if pred.min() >= 0 else torch.clamp((pred + 1) / 2, 0, 1)
        target = torch.clamp(target, 0, 1) if target.min() >= 0 else torch.clamp((target + 1) / 2, 0, 1)

        components = {}

        # 1. LPIPS (requires [-1, 1] range)
        if self.use_lpips:
            pred_lpips = pred * 2 - 1  # [0,1] -> [-1,1]
            target_lpips = target * 2 - 1
            with torch.no_grad():
                loss_lpips = self.lpips_net(pred_lpips, target_lpips).mean()
            components['lpips'] = loss_lpips.item()
        else:
            loss_lpips = torch.tensor(0.0, device=pred.device)
            components['lpips'] = 0.0

        # 2. DISTS (requires [0, 1] range)
        if self.use_dists:
            with torch.no_grad():
                loss_dists = self.dists_net(pred, target, require_grad=False).mean()
            components['dists'] = loss_dists.item()
        else:
            loss_dists = torch.tensor(0.0, device=pred.device)
            components['dists'] = 0.0

        # 3. Style loss (trainable)
        loss_style = self.compute_style_loss(pred, target)
        components['style'] = loss_style.item()

        # Weighted combination
        total_loss = (
            self.w_lpips * loss_lpips +
            self.w_dists * loss_dists +
            self.w_style * loss_style
        )

        components['total'] = total_loss.item()

        return total_loss, components


# ==============================================================================
# P0 Optimization: Retinex-Semantic Fusion Loss
# ==============================================================================

class RetinexLoss(nn.Module):
    """
    Physical constraint losses for Retinex decomposition
    Enforces I = R * L decomposition with smoothness and consistency
    """
    def __init__(self, weight_smooth=0.08, weight_consist=0.05, weight_recon=0.05):
        super().__init__()
        self.weight_smooth = weight_smooth
        self.weight_consist = weight_consist
        self.weight_recon = weight_recon
        self.eps = 1e-6

    def illumination_smoothness_loss(self, L):
        """
        Total Variation loss for illumination smoothness
        Enforces spatially smooth illumination map
        """
        # Horizontal gradients
        loss_h = torch.mean(torch.abs(L[:, :, 1:, :] - L[:, :, :-1, :]))
        # Vertical gradients
        loss_w = torch.mean(torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1]))
        return loss_h + loss_w

    def reflectance_consistency_loss(self, R, img):
        """
        Reflectance should preserve image structure
        Gradient consistency between R and input image
        """
        # Horizontal gradients
        grad_R_h = torch.abs(R[:, :, 1:, :] - R[:, :, :-1, :])
        grad_img_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        loss_h = F.l1_loss(grad_R_h, grad_img_h)

        # Vertical gradients
        grad_R_w = torch.abs(R[:, :, :, 1:] - R[:, :, :, :-1])
        grad_img_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        loss_w = F.l1_loss(grad_R_w, grad_img_w)

        return loss_h + loss_w

    def reconstruction_consistency_loss(self, R, L, img):
        """
        Reconstruction loss: I = R * L
        Ensures decomposition can reconstruct input
        """
        reconstructed = R * L.expand_as(R)
        return F.l1_loss(reconstructed, img)

    def forward(self, R, L, img):
        """
        Compute Retinex decomposition loss

        Args:
            R: Reflectance map (B, 3, H, W), range [0, 1]
            L: Illumination map (B, 1, H, W), range [0, 1]
            img: Original low-light image (B, 3, H, W)

        Returns:
            total_loss: Weighted sum of physical constraint losses
            components: Dictionary with individual loss values
        """
        loss_smooth = self.illumination_smoothness_loss(L)
        loss_consist = self.reflectance_consistency_loss(R, img)
        loss_recon = self.reconstruction_consistency_loss(R, L, img)

        total_loss = (
            self.weight_smooth * loss_smooth +
            self.weight_consist * loss_consist +
            self.weight_recon * loss_recon
        )

        components = {
            'smooth': loss_smooth.item(),
            'consist': loss_consist.item(),
            'recon': loss_recon.item(),
            'total': total_loss.item()
        }

        return total_loss, components


# ==============================================================================
# P1 Optimization: Curve Adjustment Loss
# ==============================================================================

class CurveLoss(nn.Module):
    """
    Regularization loss for curve parameters
    Prevents extreme curve adjustments and ensures smooth curves
    """
    def __init__(self, weight_tv=0.02, weight_mono=0.01):
        super().__init__()
        self.weight_tv = weight_tv
        self.weight_mono = weight_mono

    def curve_smoothness_loss(self, curve_params):
        """
        Total variation loss for curve parameters
        Encourages smooth curves without sharp transitions

        Args:
            curve_params: (B, C, N) where N is number of control points
        """
        # Difference between adjacent control points
        diff = curve_params[:, :, 1:] - curve_params[:, :, :-1]
        return torch.mean(torch.abs(diff))

    def monotonicity_loss(self, curve_params):
        """
        Encourages monotonic increasing curves
        Prevents tone reversals
        """
        diff = curve_params[:, :, 1:] - curve_params[:, :, :-1]
        # Penalize negative differences (decreasing segments)
        violation = torch.relu(-diff)
        return torch.mean(violation)

    def forward(self, curve_params):
        """
        Compute curve regularization loss

        Args:
            curve_params: (B, C, N) curve control points

        Returns:
            total_loss: Weighted regularization loss
            components: Dictionary with individual loss values
        """
        loss_tv = self.curve_smoothness_loss(curve_params)
        loss_mono = self.monotonicity_loss(curve_params)

        total_loss = (
            self.weight_tv * loss_tv +
            self.weight_mono * loss_mono
        )

        components = {
            'smoothness': loss_tv.item(),
            'monotonicity': loss_mono.item(),
            'total': total_loss.item()
        }

        return total_loss, components


# ==============================================================================
# Original CombinedLoss with P0+P1 Integration
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss function for Stage 2 training
    Integrates with actual model architecture
    Now includes P0+P1 optimizations for TIP submission
    """
    def __init__(self, config):
        super().__init__()
        
        # Initialize individual losses
        color_cfg = config['loss']['color_loss']
        self.color_loss = ColorLoss(
            weight_exp=color_cfg['weight_exp'],
            weight_color=color_cfg['weight_color'],
            weight_tv=color_cfg['weight_tv']
        )
        self.semantic_loss = SemanticConsistencyLoss()
        self.freq_loss = FrequencyLoss()
        self.noise_loss = NoiseSuppressionLoss()
        self.perceptual_loss = PerceptualLoss()
        self.task_loss = TaskAlignmentLoss()
        self.ssim_loss = SSIMLoss()  # Added for validation
        self.recon_loss = nn.L1Loss()

        # ==== P0+P1 Optimizations ====
        # Advanced perceptual loss (LPIPS + DISTS + Style)
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        use_advanced_perceptual = config.get('model', {}).get('use_advanced_perceptual', True)
        if use_advanced_perceptual:
            self.advanced_perceptual = AdvancedPerceptualLoss(device=device)
            print("✓ Enabled Advanced Perceptual Loss (LPIPS + DISTS + Style)")
        else:
            self.advanced_perceptual = None

        # Retinex decomposition loss
        use_retinex = config.get('model', {}).get('retinex_semantic', {}).get('enabled', False)
        if use_retinex:
            retinex_cfg = config['model']['retinex_semantic']
            self.retinex_loss = RetinexLoss(
                weight_smooth=retinex_cfg.get('weight_smooth', 0.08),
                weight_consist=retinex_cfg.get('weight_consist', 0.05),
                weight_recon=retinex_cfg.get('weight_recon', 0.05)
            )
            print("✓ Enabled Retinex-Semantic Fusion Loss")
        else:
            self.retinex_loss = None

        # Curve adjustment loss
        use_curve = config.get('model', {}).get('curve_transformer', {}).get('enabled', False)
        if use_curve:
            curve_cfg = config['model'].get('curve_transformer', {})
            self.curve_loss = CurveLoss(
                weight_tv=curve_cfg.get('weight_tv', 0.02),
                weight_mono=curve_cfg.get('weight_mono', 0.01)
            )
            print("✓ Enabled Curve Adjustment Loss")
        else:
            self.curve_loss = None
        
        # Loss weights (will be updated by curriculum)
        self.weights = {
            'color': config['loss'].get('w_color', 0.0001),
            'semantic': config['loss'].get('w_semantic', 0.5),
            'freq': config['loss'].get('w_freq', 0.0001),
            'noise': config['loss'].get('w_noise', 0.0),
            'perceptual': config['loss'].get('w_perceptual', 0.5),
            'task': config['loss'].get('w_task', 0.3),
            'adv': config['loss'].get('w_adv', 0.0),
            'recon': config['loss'].get('w_recon', 0.0),
            'brightness': config['loss'].get('w_brightness', 0.0),
            # P0+P1 optimization weights
            'advanced_perceptual': config['loss'].get('w_advanced_perceptual', 0.30),
            'retinex': config['loss'].get('w_retinex', 0.10),
            'curve': config['loss'].get('w_curve', 0.02)
        }
        
        # Curriculum configuration
        self.curriculum_config = config.get('curriculum', {})
        self.enabled_curriculum = self.curriculum_config.get('enabled', False)
        self.eps = 1e-8

        self.curriculum = config.get('curriculum', {})
        self.phase_defs = {}

        for k, v in self.curriculum.items():
            if k.startswith('phase') and 'epochs' in v:
                self.phase_defs[k] = v

        self._last_weights = {}

    def get_last_weights(self):
        return getattr(self, '_last_weights', {}).copy()

    def _phase_for_epoch(self, epoch):
        if not self.phase_defs:
            return None
        for name, spec in self.phase_defs.items():
            start, end = spec['epochs']
            if start <= epoch <= end:
                return name
        # fallback to last defined phase
        return list(self.phase_defs.keys())[-1]

    def _interp(self, val, epoch, phase):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, (list, tuple)) and phase in self.phase_defs:
            start, end = self.phase_defs[phase]['epochs']
            duration = max(1, (end - start))
            t = (epoch - start) / duration
            t = max(0.0, min(1.0, t))
            return float(val[0] + (val[1] - val[0]) * t)
        return 0.0

    def _pull_phase_weights(self, epoch):
        phase = self._phase_for_epoch(epoch)
        if phase is None:
            self._last_weights = {}
            return self._last_weights
        loss_weights = self.phase_defs.get(phase, {}).get('loss_weights', {})
        weights = {
            'phase': phase,
            'w_color': self._interp(loss_weights.get('w_color', 0.0), epoch, phase),
            'w_semantic': self._interp(loss_weights.get('w_semantic', 0.0), epoch, phase),
            'w_perceptual': self._interp(loss_weights.get('w_perceptual', 0.0), epoch, phase),
            'w_adv': self._interp(loss_weights.get('w_adv', 0.0), epoch, phase),
            'w_freq': self._interp(loss_weights.get('w_freq', 0.0), epoch, phase),
            'w_noise': self._interp(loss_weights.get('w_noise', 0.0), epoch, phase),
            'w_recon': self._interp(loss_weights.get('w_recon', 0.0), epoch, phase),
            'w_brightness': self._interp(loss_weights.get('w_brightness', 0.0), epoch, phase),
        }
        self._last_weights = weights
        return weights

    def update_weights(self, weights_dict):
        """Update loss weights (called by curriculum scheduler)"""
        for key, value in weights_dict.items():
            mapped_key = key
            if mapped_key.startswith('w_'):
                mapped_key = mapped_key[2:]
            if mapped_key in self.weights:
                self.weights[mapped_key] = value
    
    def get_curriculum_weights(self, epoch):
        """
        Get loss weights for current epoch based on curriculum
        
        Args:
            epoch: Current training epoch
        
        Returns:
            Dictionary of loss weights
        """
        if not self.enabled_curriculum:
            return self.weights.copy()
        
        # Determine current phase
        phase_configs = {
            1: self.curriculum_config.get('phase1', {}),
            2: self.curriculum_config.get('phase2', {}),
            3: self.curriculum_config.get('phase3', {}),
            4: self.curriculum_config.get('phase4', {}),
            5: self.curriculum_config.get('phase5', {}),
            6: self.curriculum_config.get('phase6', {})
        }
        
        current_phase = None
        for phase, phase_cfg in phase_configs.items():
            epoch_range = phase_cfg.get('epochs', [1, 1])
            if epoch >= epoch_range[0] and epoch <= epoch_range[1]:
                current_phase = phase
                break
        
        if current_phase is None:
            return self.weights.copy()
        
        # Get phase configuration
        phase_cfg = phase_configs[current_phase]
        loss_weights = phase_cfg.get('loss_weights', {})
        
        # Interpolate weights if they are lists [start, end]
        epoch_range = phase_cfg['epochs']
        start_epoch, end_epoch = epoch_range[0], epoch_range[1]
        progress = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
        
        current_weights = {}
        for key, value in loss_weights.items():
            if isinstance(value, list) and len(value) == 2:
                # Linear interpolation
                current_weights[key] = value[0] + progress * (value[1] - value[0])
            else:
                current_weights[key] = value
        
        return current_weights
    
    def forward(self, enhanced, high, low, epoch=1, disc_pred=None, retinex_components=None, curve_params=None):
        """
        Compute combined loss with P0+P1 optimizations

        Args:
            enhanced: Enhanced image from generator
            high: High-quality target image
            low: Low-light input image
            epoch: Current epoch (for curriculum)
            disc_pred: Discriminator prediction (optional)
            retinex_components: Dict with 'R' and 'L' from Retinex decomposition (optional)
            curve_params: Curve control points (B, C, N) (optional)

        Returns:
            Dictionary of losses
        """
        W = self._pull_phase_weights(epoch)
        # Update weights based on curriculum
        if W:
            phase_weights = {k[2:]: v for k, v in W.items() if k.startswith('w_')}
            if phase_weights:
                self.update_weights(phase_weights)
        if self.enabled_curriculum:
            current_weights = self.get_curriculum_weights(epoch)
            if current_weights:
                self.update_weights(current_weights)

        losses = {}
        total_loss = torch.tensor(0.0, device=enhanced.device, requires_grad=True)

        # ========== P0 Optimization: Advanced Perceptual Loss ==========
        if self.advanced_perceptual is not None and self.weights.get('advanced_perceptual', 0) > 0:
            try:
                adv_perc_loss, adv_perc_components = self.advanced_perceptual(enhanced, high)
                if not torch.isnan(adv_perc_loss).any() and not torch.isinf(adv_perc_loss).any():
                    losses['advanced_perceptual'] = adv_perc_loss
                    losses['adv_perc_components'] = adv_perc_components
                    total_loss = total_loss + self.weights['advanced_perceptual'] * adv_perc_loss
            except Exception as e:
                print(f"Warning: Advanced perceptual loss failed: {e}")
                pass

        # ========== P0 Optimization: Retinex Physical Constraints ==========
        if self.retinex_loss is not None and retinex_components is not None and self.weights.get('retinex', 0) > 0:
            try:
                R = retinex_components.get('R')
                L = retinex_components.get('L')
                if R is not None and L is not None:
                    retinex_loss_val, retinex_comps = self.retinex_loss(R, L, low)
                    if not torch.isnan(retinex_loss_val).any() and not torch.isinf(retinex_loss_val).any():
                        losses['retinex'] = retinex_loss_val
                        losses['retinex_components'] = retinex_comps
                        total_loss = total_loss + self.weights['retinex'] * retinex_loss_val
            except Exception as e:
                print(f"Warning: Retinex loss failed: {e}")
                pass

        # ========== P1 Optimization: Curve Regularization ==========
        if self.curve_loss is not None and curve_params is not None and self.weights.get('curve', 0) > 0:
            try:
                curve_loss_val, curve_comps = self.curve_loss(curve_params)
                if not torch.isnan(curve_loss_val).any() and not torch.isinf(curve_loss_val).any():
                    losses['curve'] = curve_loss_val
                    losses['curve_components'] = curve_comps
                    total_loss = total_loss + self.weights['curve'] * curve_loss_val
            except Exception as e:
                print(f"Warning: Curve loss failed: {e}")
                pass
        
        # Color loss (always enabled)
        if self.weights.get('color', 0) > 0:
            try:
                color_l = self.color_loss(enhanced, high)
                if not torch.isnan(color_l).any() and not torch.isinf(color_l).any():
                    losses['color'] = color_l
                    total_loss = total_loss + self.weights['color'] * color_l
            except Exception:
                pass
        
        # Perceptual loss
        if self.weights.get('perceptual', 0) > 0:
            try:
                perc_l = self.perceptual_loss(enhanced, high)
                if not torch.isnan(perc_l).any() and not torch.isinf(perc_l).any():
                    losses['perceptual'] = perc_l
                    total_loss = total_loss + self.weights['perceptual'] * perc_l
            except Exception:
                pass

        # Reconstruction loss (L1)
        if self.weights.get('recon', 0) > 0:
            try:
                recon_l = self.recon_loss(enhanced, high)
                if not torch.isnan(recon_l).any() and not torch.isinf(recon_l).any():
                    losses['recon'] = recon_l
                    total_loss = total_loss + self.weights['recon'] * recon_l
            except Exception:
                pass
        
        # Frequency loss
        if self.weights.get('freq', 0) > 0:
            try:
                freq_l = self.freq_loss(enhanced, high)
                if not torch.isnan(freq_l).any() and not torch.isinf(freq_l).any():
                    losses['freq'] = freq_l
                    total_loss = total_loss + self.weights['freq'] * freq_l
            except Exception:
                pass
        
        # Noise suppression loss
        if self.weights.get('noise', 0) > 0:
            try:
                noise_l = self.noise_loss(enhanced, high)
                if not torch.isnan(noise_l).any() and not torch.isinf(noise_l).any():
                    losses['noise'] = noise_l
                    total_loss = total_loss + self.weights['noise'] * noise_l
            except Exception:
                pass
        
        # Brightness alignment loss
        if self.weights.get('brightness', 0) > 0:
            try:
                bright_l = torch.mean((enhanced.mean(dim=[2, 3]) - high.mean(dim=[2, 3])) ** 2)
                if not torch.isnan(bright_l).any() and not torch.isinf(bright_l).any():
                    losses['brightness'] = bright_l
                    total_loss = total_loss + self.weights['brightness'] * bright_l
            except Exception:
                pass
        
        # Adversarial loss
        if self.weights.get('adv', 0) > 0 and disc_pred is not None:
            try:
                # Generator wants discriminator to output 1 (real)
                adv_labels = torch.ones_like(disc_pred)
                adv_l = F.binary_cross_entropy_with_logits(disc_pred, adv_labels)
                if not torch.isnan(adv_l).any() and not torch.isinf(adv_l).any():
                    losses['adv'] = adv_l
                    total_loss = total_loss + self.weights['adv'] * adv_l
            except Exception:
                pass
        
       # === Semantic loss ===
        if self.weights.get('semantic', 0) > 0 and hasattr(self, 'semantic_loss'):
            try:
                sem_l = self.semantic_loss(enhanced, high)
                if not torch.isnan(sem_l).any() and not torch.isinf(sem_l).any():
                    losses['semantic'] = sem_l
                    total_loss = total_loss + self.weights['semantic'] * sem_l
            except Exception:
                # if semantic head not used in early phase, return zero-safe
                losses['semantic'] = 0.0 * enhanced.mean()
        # === Clamp total & return ===
        total_loss = torch.clamp(total_loss, 0, 50.0)
        losses['total'] = total_loss
        return losses

if __name__ == '__main__':
    # Test losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy config
    config = {
        'loss': {
            'w_color': 0.001,
            'w_semantic': 0.5,
            'w_freq': 0.0001,
            'w_perceptual': 0.5,
            'w_task': 0.3,
            'w_adv': 0.0,
            'color_loss': {
                'weight_exp': 10,
                'weight_color': 5,
                'weight_tv': 200
            }
        },
        'curriculum': {
            'enabled': True,
            'phase1': {
                'epochs': [1, 20],
                'loss_weights': {'w_color': 0.001, 'w_semantic': 0.0}
            }
        }
    }
    
    # Test combined loss
    criterion = CombinedLoss(config).to(device)
    
    enhanced = torch.randn(2, 3, 256, 256).to(device)
    high = torch.randn(2, 3, 256, 256).to(device)
    low = torch.randn(2, 3, 256, 256).to(device)
    
    losses = criterion(enhanced, high, low, epoch=1)
    
    
    print("Loss computation successful!")
    print(f"Total loss: {losses['total'].item():.4f}")
    for key, value in losses.items():
        if key != 'total' and isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
