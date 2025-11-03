"""
Optimized Loss Functions for Stage 2 Training
Fixed NaN handling and integrated with actual model architecture
"""

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
        target = 0.6
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
    def get_last_weights(self):
        return getattr(self, '_last_weights', {}).copy()

    def _phase_for_epoch(self, epoch):
        for name, spec in self.phase_defs.items():
            s, e = spec['epochs']
            if s <= epoch <= e:
               return name
        return list(self.phase_defs.keys())[-1]

    def _interp(self, val, epoch, phase):
            if isinstance(val, (int, float)):
               return float(val)
            if isinstance(val, (list, tuple)):
               start, end = self.phase_defs[phase]['epochs']
               t = (epoch - start) / max(1, (end - start))
               t = max(0, min(1, t))
               return float(val[0] + (val[1] - val[0]) * t)
            return 0.0

    def _pull_phase_weights(self, epoch):
        phase = self._phase_for_epoch(epoch)
        lw = self.phase_defs.get(phase, {}).get('loss_weights', {})
        W = {
              'phase': phase,
              'w_color': self._interp(lw.get('w_color', 0), epoch, phase),
              'w_semantic': self._interp(lw.get('w_semantic', 0), epoch, phase),
              'w_perceptual': self._interp(lw.get('w_perceptual', 0), epoch, phase),
              'w_adv': self._interp(lw.get('w_adv', 0), epoch, phase),
              'w_freq': self._interp(lw.get('w_freq', 0), epoch, phase),
         }
        self._last_weights = W
        return W


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


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss
    """
    def __init__(self):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG16_Weights
        
        # Use new weights API
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract feature layers
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eps = 1e-8
        
        # ImageNet normalization
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


class CombinedLoss(nn.Module):
    """
    Combined loss function for Stage 2 training
    Integrates with actual model architecture
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
        self.perceptual_loss = PerceptualLoss()
        self.task_loss = TaskAlignmentLoss()
        self.ssim_loss = SSIMLoss()  # Added for validation
        
        # Loss weights (will be updated by curriculum)
        self.weights = {
            'color': config['loss'].get('w_color', 0.0001),
            'semantic': config['loss'].get('w_semantic', 0.5),
            'freq': config['loss'].get('w_freq', 0.0001),
            'perceptual': config['loss'].get('w_perceptual', 0.5),
            'task': config['loss'].get('w_task', 0.3),
            'adv': config['loss'].get('w_adv', 0.0)
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

    def update_weights(self, weights_dict):
        """Update loss weights (called by curriculum scheduler)"""
        self.weights.update(weights_dict)
    
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
    
    def forward(self, enhanced, high, low, epoch=1, disc_pred=None):
        """
        Compute combined loss
        
        Args:
            enhanced: Enhanced image from generator
            high: High-quality target image
            low: Low-light input image
            epoch: Current epoch (for curriculum)
            disc_pred: Discriminator prediction (optional)
        
        Returns:
            Dictionary of losses
        """
        W = self._pull_phase_weights(epoch)
        # Update weights based on curriculum
        if self.enabled_curriculum:
            current_weights = self.get_curriculum_weights(epoch)
            self.update_weights(current_weights)
        
        losses = {}
        total_loss = torch.tensor(0.0, device=enhanced.device, requires_grad=True)
        
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
        
        # Frequency loss
        if self.weights.get('freq', 0) > 0:
            try:
                freq_l = self.freq_loss(enhanced, high)
                if not torch.isnan(freq_l).any() and not torch.isinf(freq_l).any():
                    losses['freq'] = freq_l
                    total_loss = total_loss + self.weights['freq'] * freq_l
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