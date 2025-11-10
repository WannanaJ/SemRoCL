"""
Optimized Stage 2 Training Script for SemRoCL with Curriculum Learning
Key improvements:
- Fixed gradient explosion with better clipping strategy
- Improved NaN handling at root causes
- Better integrated with actual model architecture
- Enhanced monitoring and stability checks
"""

import os
import sys
import yaml
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings("ignore")

try:
    from torch.amp import GradScaler  # type: ignore[attr-defined]
    _GRADSCALER_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore
    _GRADSCALER_NEW_API = False
from torch.cuda.amp import autocast

# Helper to create GradScaler compatible with both APIs
def create_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if _GRADSCALER_NEW_API:
        return GradScaler(enabled=True)
    return GradScaler(enabled=True)

# Import custom modules with error handling
try:
    from data_loader import get_dataloader
except ImportError as e:
    raise ImportError(f"Cannot import data_loader: {e}. Please ensure data_loader.py exists.")

try:
    from loss_functions import CombinedLoss
except ImportError as e:
    raise ImportError(f"Cannot import loss_functions: {e}. Please ensure loss_functions.py exists.")

try:
    from model.models import EnhancementGenerator, Discriminator
except ImportError as e:
    raise ImportError(f"Cannot import models: {e}. Please ensure models.py exists.")


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Stage2Trainer:
    """Optimized curriculum-based training for Stage 2"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizers and schedulers
        self.setup_optimizers()
        
        # Initialize loss function
        self.criterion = CombinedLoss(config).to(self.device)
        
        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler_g = create_grad_scaler(self.use_amp)
        self.scaler_d = create_grad_scaler(self.use_amp)
        if self.use_amp and (self.scaler_g is None or self.scaler_d is None):
            raise RuntimeError("AMP is enabled but GradScaler could not be initialized.")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Curriculum tracking
        self.current_phase = 1
        
        # Setup logging
        self.setup_logging()
        
        # Gradient accumulation
        self.accumulate_steps = config['training'].get('accumulate_steps', 1)
        
        # FIXED: Separate gradient clipping for generator and discriminator
        self.gradient_clip_g = config['training'].get('gradient_clip', 1.0)
        self.gradient_clip_d = config['training'].get('gradient_clip_discriminator', 5.0)
        
        # Discriminator schedule
        self.disc_start_epoch = config['training'].get('disc_start_epoch', 90)
        self.disc_update_freq = config['training'].get('disc_update_freq', 5)
        
        # EMA
        if config['training'].get('use_ema', False):
            self.ema = EMA(self.generator, decay=config['training'].get('ema_decay', 0.999))
        else:
            self.ema = None
        
        # ADDED: Gradient explosion detection
        self.max_grad_norm = config['monitoring'].get('max_grad_norm', 10.0)
        self.gradient_explosion_count = 0
        self.max_gradient_explosions = 5
        
        print(f"Trainer initialized on {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def setup_models(self):
        """Initialize generator and discriminator"""
        model_cfg = self.config['model']
        
        # Generator configuration
        gen_config = {
            'enhancer_channels': model_cfg.get('enhancer_channels', 32),
            'num_iterations': model_cfg.get('num_iterations', 8),
            'semantic_channels': model_cfg.get('semantic_channels', 64),
            'use_semantic': model_cfg.get('use_semantic', True)
        }
        
        self.generator = EnhancementGenerator(gen_config).to(self.device)
        
        # Load pretrained encoder if available
        pretrained_path = model_cfg.get('pretrained_encoder')
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
                # Try to load encoder weights
                if 'encoder_state_dict' in checkpoint:
                    # FIXED: Load only matching weights with shape validation
                    state_dict = checkpoint['encoder_state_dict']
                    model_dict = self.generator.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    self.generator.load_state_dict(model_dict)
                    print(f"Loaded pretrained encoder ({len(pretrained_dict)} layers)")
            except Exception as e:
                print(f"Warning: Could not load pretrained encoder: {e}")
        
        # Discriminator
        self.discriminator = Discriminator(
            in_channels=3,
            ndf=64,
            n_layers=3,
            use_spectral_norm=True
        ).to(self.device)
        
        # Print model info
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers"""
        train_cfg = self.config['training']
        
        # Generator optimizer
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=train_cfg.get('lr_generator', 0.0001),
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Discriminator optimizer
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=train_cfg.get('lr_discriminator', 0.000005),
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate schedulers
        if train_cfg.get('use_cosine_schedule', True):
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g,
                T_max=train_cfg['epochs'],
                eta_min=1e-7
            )
            
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d,
                T_max=train_cfg['epochs'],
                eta_min=1e-8
            )
        else:
            # Step scheduler as alternative
            milestones = [60, 120, 170]
            self.scheduler_g = optim.lr_scheduler.MultiStepLR(
                self.optimizer_g,
                milestones=milestones,
                gamma=0.5
            )
            
            self.scheduler_d = optim.lr_scheduler.MultiStepLR(
                self.optimizer_d,
                milestones=milestones,
                gamma=0.5
            )
        
        print("Optimizers initialized")
        print(f"  Generator LR: {train_cfg.get('lr_generator', 0.0001)}")
        print(f"  Discriminator LR: {train_cfg.get('lr_discriminator', 0.000005)}")
    
    def setup_logging(self):
        """Setup logging directories and writers"""
        exp_name = self.config['exp_name']
        output_dir = self.config['output_dir']
        
        self.output_dir = os.path.join(output_dir, exp_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.image_dir = os.path.join(self.output_dir, 'images')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # CSV logger
        self.csv_path = os.path.join(self.log_dir, 'training_log.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'gen_loss', 'disc_loss',
                'color_loss', 'semantic_loss', 'perceptual_loss', 
                'adv_loss', 'freq_loss', 'lr', 'w_semantic'
            ])
        
        print(f"Logging initialized at {self.output_dir}")
    
    def get_current_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch based on curriculum"""
        curriculum_cfg = self.config.get('curriculum', {})
        
        if not curriculum_cfg.get('enabled', False):
            return self.config['loss']
        
        # Determine current phase
        for phase_num in range(1, 7):
            phase_key = f'phase{phase_num}'
            if phase_key not in curriculum_cfg:
                continue
            
            phase = curriculum_cfg[phase_key]
            epoch_range = phase['epochs']
            
            if epoch_range[0] <= epoch <= epoch_range[1]:
                self.current_phase = phase_num
                weights = phase['loss_weights'].copy()
                
                # Interpolate weights if they are ranges
                for key, value in weights.items():
                    if isinstance(value, list) and len(value) == 2:
                        # Linear interpolation
                        start_val, end_val = value
                        progress = (epoch - epoch_range[0]) / (epoch_range[1] - epoch_range[0])
                        weights[key] = start_val + (end_val - start_val) * progress
                
                return weights
        
        # Default to base loss weights
        return self.config['loss']
    
    def clip_gradients_safely(self, model, max_norm: float, model_name: str = "Model") -> float:
        """
        IMPROVED: Clip gradients with better monitoring and explosion detection
        Returns the total gradient norm before clipping
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Calculate total norm
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float('inf'))
        
        # ADDED: Check for gradient explosion
        if total_norm > self.max_grad_norm * 100:  # 100x threshold
            self.gradient_explosion_count += 1
            print(f"WARNING: Gradient explosion detected in {model_name}! Norm: {total_norm:.2f}")
            
            if self.gradient_explosion_count > self.max_gradient_explosions:
                print(f"ERROR: Too many gradient explosions ({self.gradient_explosion_count}). Stopping training.")
                raise RuntimeError("Training unstable: too many gradient explosions")
        
        # Clip gradients
        if total_norm > max_norm:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        
        return total_norm.item()
    
    def train_epoch(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch with improved stability"""
        self.generator.train()
        self.discriminator.train()
        
        # Get current loss weights
        current_weights = self.get_current_loss_weights(epoch)
        self.criterion.update_weights(current_weights)
        
        # Track losses
        epoch_losses = {
            'gen_total': 0.0, 'disc_total': 0.0,
            'color': 0.0, 'semantic': 0.0, 'perceptual': 0.0,
            'adv': 0.0, 'freq': 0.0
        }
        
        num_batches = 0
        current_lr = self.optimizer_g.param_groups[0]['lr']
        current_w_semantic = current_weights.get('w_semantic', 0.0)
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config['training']['epochs']} [Phase {self.current_phase}]")
        
        for batch_idx, batch in enumerate(pbar):
            low_img = batch['low'].to(self.device)
            high_img = batch['high'].to(self.device)
            
            # ADDED: Input validation
            if torch.isnan(low_img).any() or torch.isnan(high_img).any():
                print(f"Warning: NaN detected in input batch {batch_idx}, skipping...")
                continue
            
            # Train Generator
            should_train_g = (batch_idx % self.accumulate_steps == 0) or (batch_idx == len(dataloader) - 1)
            
            if should_train_g:
                self.optimizer_g.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                enhanced = self.generator(low_img)
                
                # ADDED: Output validation
                if torch.isnan(enhanced).any():
                    print(f"Warning: NaN in generator output at batch {batch_idx}, skipping...")
                    continue
                
                # Compute generator losses
                losses_g = self.criterion(enhanced, high_img, low_img)
                
                # ADDED: Loss validation
                if torch.isnan(losses_g['total']):
                    print(f"Warning: NaN in loss at batch {batch_idx}, skipping...")
                    continue
                
                # Adversarial loss (if discriminator is active)
                if epoch >= self.disc_start_epoch and batch_idx % self.disc_update_freq == 0:
                    fake_pred = self.discriminator(enhanced)
                    adv_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, 
                        torch.ones_like(fake_pred) * 0.9
                    )
                    losses_g['adv'] = adv_loss
                    losses_g['total'] = losses_g['total'] + current_weights.get('w_adv', 0.0) * adv_loss
            
            # Backward pass for generator
            if self.use_amp:
                if self.scaler_g is None:
                    raise RuntimeError("GradScaler expected but not initialized for generator.")
                self.scaler_g.scale(losses_g['total']).backward()
            else:
                losses_g['total'].backward()
            
            # Update generator
            if should_train_g:
                if self.use_amp:
                    if self.scaler_g is None:
                        raise RuntimeError("GradScaler expected but not initialized for generator.")
                    self.scaler_g.unscale_(self.optimizer_g)
                    grad_norm_g = self.clip_gradients_safely(
                        self.generator,
                        self.gradient_clip_g,
                        "Generator"
                    )
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                else:
                    grad_norm_g = self.clip_gradients_safely(
                        self.generator,
                        self.gradient_clip_g,
                        "Generator"
                    )
                    self.optimizer_g.step()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
            
            # Train Discriminator
            disc_loss = torch.tensor(0.0).to(self.device)
            if epoch >= self.disc_start_epoch and batch_idx % self.disc_update_freq == 0:
                self.optimizer_d.zero_grad()
                
                with autocast(enabled=self.use_amp):
                    # Real images
                    real_pred = self.discriminator(high_img)
                    real_loss = F.binary_cross_entropy_with_logits(
                        real_pred, 
                        torch.ones_like(real_pred) * 0.9
                    )
                    
                    # Fake images (detached)
                    fake_pred = self.discriminator(enhanced.detach())
                    fake_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, 
                        torch.zeros_like(fake_pred) + 0.1
                    )
                    
                    disc_loss = (real_loss + fake_loss) * 0.5
                
                if self.use_amp:
                    if self.scaler_d is None:
                        raise RuntimeError("GradScaler expected but not initialized for discriminator.")
                    self.scaler_d.scale(disc_loss).backward()
                    self.scaler_d.unscale_(self.optimizer_d)
                    grad_norm_d = self.clip_gradients_safely(
                        self.discriminator,
                        self.gradient_clip_d,
                        "Discriminator"
                    )
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    disc_loss.backward()
                    grad_norm_d = self.clip_gradients_safely(
                        self.discriminator,
                        self.gradient_clip_d,
                        "Discriminator"
                    )
                    self.optimizer_d.step()
            
            # Accumulate losses
            epoch_losses['gen_total'] += losses_g['total'].item()
            epoch_losses['disc_total'] += disc_loss.item()
            for key in ['color', 'semantic', 'perceptual', 'adv', 'freq']:
                if key in losses_g:
                    epoch_losses[key] += losses_g[key].item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{losses_g['total'].item():.4f}",
                'D': f"{disc_loss.item():.4f}",
                'LR': f"{current_lr:.6f}",
                'Phase': self.current_phase
            })
            
            # Log to tensorboard
            if self.global_step % self.config['monitoring'].get('print_freq', 50) == 0:
                for key in ['total', 'color', 'semantic', 'perceptual', 'adv', 'freq']:
                    if key in losses_g:
                        self.writer.add_scalar(f'Loss/{key}', losses_g[key].item(), self.global_step)
            
            # Log to CSV
            if self.global_step % 50 == 0:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, batch_idx, self.current_phase,
                        losses_g['total'].item(), disc_loss.item(),
                        losses_g.get('color', torch.tensor(0)).item(),
                        losses_g.get('semantic', torch.tensor(0)).item(),
                        losses_g.get('perceptual', torch.tensor(0)).item(),
                        losses_g.get('adv', torch.tensor(0)).item(),
                        losses_g.get('freq', torch.tensor(0)).item(),
                        current_lr, current_w_semantic
                    ])
            
            # Save sample images
            if self.global_step % self.config['monitoring'].get('save_images_freq', 1000) == 0:
                self.save_sample_images(low_img, enhanced, high_img, epoch)
            
            self.global_step += 1
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.generator.eval()
        
        # Apply EMA if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        val_losses = {'total': 0.0, 'psnr': 0.0, 'ssim': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                low_img = batch['low'].to(self.device)
                high_img = batch.get('high')
                
                if high_img is None:
                    continue
                
                high_img = high_img.to(self.device)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    enhanced = self.generator(low_img)
                    
                    # Compute metrics
                    mse = F.mse_loss(enhanced, high_img)
                    val_losses['total'] += mse.item()
                    
                    # PSNR
                    psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
                    val_losses['psnr'] += psnr.item()
                    
                    # SSIM (if available in criterion)
                    if hasattr(self.criterion, 'ssim_loss'):
                        ssim_val = self.criterion.ssim_loss(enhanced, high_img)
                        val_losses['ssim'] += (1 - ssim_val.item())
                    else:
                        val_losses['ssim'] += 0.0
                
                num_batches += 1
        
        # Restore original parameters
        if self.ema is not None:
            self.ema.restore()
        
        # Average metrics
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_losses['total'], epoch)
        self.writer.add_scalar('Val/PSNR', val_losses['psnr'], epoch)
        self.writer.add_scalar('Val/SSIM', val_losses['ssim'], epoch)
        
        self.generator.train()
        
        return val_losses
    
    def check_training_health(self, epoch: int):
        """Check training health and print diagnostics"""
        print("\n" + "="*70)
        print(f" Training Health Check - Epoch {epoch}")
        print("="*70)
        
        # Check generator gradients
        gen_grads = []
        for name, param in self.generator.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gen_grads.append(grad_norm)
                if grad_norm > 100:  # FIXED: Lower threshold
                    print(f"Warning: Large gradient in {name}: {grad_norm:.2f}")
        
        if len(gen_grads) > 0:
            print(f"  Gradient norm: mean={np.mean(gen_grads):.4f}, max={np.max(gen_grads):.4f}")
        
        # Check parameter norms
        param_norms = []
        for param in self.generator.parameters():
            param_norms.append(param.norm().item())
        
        if len(param_norms) > 0:
            print(f"  Param norm: mean={np.mean(param_norms):.4f}, max={np.max(param_norms):.4f}")
        
        print("="*70 + "\n")
    
    def save_sample_images(self, low, enhanced, target, epoch):
        """
        Save sample images for visualization with artifact reduction

        保存样本图像，应用降噪和后处理以减少伪影
        Save sample images with denoising and post-processing to reduce artifacts
        """
        import torchvision.utils as vutils

        # 对增强图像进行后处理，减少噪点和伪影
        # Post-process enhanced image to reduce noise and artifacts
        with torch.no_grad():
            enhanced_processed = enhanced.clone()

            # 应用轻微的高斯平滑减少噪点（在 tensor 层面）
            # Apply slight Gaussian smoothing to reduce noise at tensor level
            kernel_size = 3
            sigma = 0.5
            kernel = self._get_gaussian_kernel(kernel_size, sigma).to(enhanced.device)

            # 对每个通道应用高斯滤波
            # Apply Gaussian filter to each channel
            for i in range(enhanced_processed.size(1)):
                enhanced_processed[:, i:i+1] = F.conv2d(
                    enhanced_processed[:, i:i+1],
                    kernel,
                    padding=kernel_size // 2
                )

            # 裁剪到有效范围
            # Clamp to valid range
            enhanced_processed = torch.clamp(enhanced_processed, 0, 1)

        # Take first image from batch
        low_img = low[0:1]
        enhanced_img = enhanced_processed[0:1]
        target_img = target[0:1]

        # Create comparison grid
        comparison = torch.cat([low_img, enhanced_img, target_img], dim=3)

        # Save image with high quality
        save_path = os.path.join(self.image_dir, f'epoch_{epoch:03d}_step_{self.global_step}.png')
        vutils.save_image(comparison, save_path, normalize=False, format='PNG')

        # 额外保存单独的高质量增强图像
        # Save separate high-quality enhanced image
        from utils import save_image
        enhanced_path = os.path.join(self.image_dir, f'epoch_{epoch:03d}_step_{self.global_step}_enhanced.png')
        save_image(enhanced_img, enhanced_path, quality=95, apply_post_processing=True)

        # Log to tensorboard
        self.writer.add_image('Images/Comparison', comparison[0], self.global_step)
        self.writer.add_image('Images/Enhanced', enhanced_img[0], self.global_step)

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float):
        """
        生成高斯核用于平滑
        Generate Gaussian kernel for smoothing
        """
        # 创建一维高斯核
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # 创建二维高斯核
        # Create 2D Gaussian kernel
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        # 重塑为卷积核格式 (out_channels, in_channels, H, W)
        # Reshape to convolution kernel format
        return kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'current_phase': self.current_phase,
            'global_step': self.global_step
        }
        
        # Save EMA if available
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save regular checkpoint
        if epoch % self.config['training'].get('save_freq', 5) == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Always save latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.current_phase = checkpoint.get('current_phase', 1)
        
        # Load EMA if available
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"Resumed from epoch {self.current_epoch}")
        return True
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        print("\n" + "="*70)
        print(" Starting Stage 2 Curriculum Training (Optimized)")
        print("="*70)
        print(f" Total epochs: {self.config['training']['epochs']}")
        print(f" Curriculum phases: 6")
        print(f" Gradient accumulation: {self.accumulate_steps} steps")
        print(f" Mixed precision: {self.use_amp}")
        print(f" EMA: {self.ema is not None}")
        print("="*70 + "\n")
        
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = get_dataloader(self.config, split='train')
        
        # Check for validation set
        val_dir = os.path.join(self.config['dataset']['root_dir'], 'val')
        if os.path.exists(val_dir):
            val_loader = get_dataloader(self.config, split='val')
            print(f"Validation set found")
        else:
            val_loader = None
            print(f"Warning: No validation set found")
        
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        
        # Training loop
        start_epoch = self.current_epoch + 1
        for epoch in range(start_epoch, self.config['training']['epochs'] + 1):
            self.current_epoch = epoch
            
            epoch_start_time = time.time()
            
            # Train one epoch
            train_losses = self.train_epoch(epoch, train_loader)
            
            # Validation
            val_losses = {'total': 0, 'psnr': 0, 'ssim': 0}
            if val_loader and epoch % 5 == 0:
                val_losses = self.validate(epoch, val_loader)
            
            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\n" + "="*70)
            print(f" Epoch {epoch} Summary [Phase {self.current_phase}] ({epoch_time:.1f}s)")
            print(f"="*70)
            print(f"  Generator Loss: {train_losses['gen_total']:.4f}")
            print(f"    |- Color: {train_losses['color']:.4f}")
            print(f"    |- Semantic: {train_losses['semantic']:.4f}")
            print(f"    |- Perceptual: {train_losses['perceptual']:.4f}")
            print(f"    |- Adversarial: {train_losses['adv']:.4f}")
            print(f"    |- Frequency: {train_losses['freq']:.4f}")
            print(f"  Discriminator Loss: {train_losses['disc_total']:.4f}")
            
            if val_loader and epoch % 5 == 0:
                print(f"  Validation:")
                print(f"    |- Loss: {val_losses['total']:.4f}")
                print(f"    |- PSNR: {val_losses['psnr']:.2f} dB")
                print(f"    |- SSIM: {val_losses['ssim']:.4f}")
            
            print(f"  Learning Rate: {self.optimizer_g.param_groups[0]['lr']:.7f}")
            print("="*70)
            
            # Check training health every 10 epochs
            if epoch % 10 == 0:
                self.check_training_health(epoch)
            
            # Save checkpoint
            is_best = False
            if val_loader:
                is_best = val_losses['total'] < self.best_metric
                if is_best:
                    self.best_metric = val_losses['total']
            
            self.save_checkpoint(epoch, val_losses['total'], is_best)
        
        # Close writer
        self.writer.close()
        
        print("\n" + "="*70)
        print(" Training completed successfully!")
        if val_loader:
            print(f" Best validation loss: {self.best_metric:.4f}")
        print(f" Checkpoints saved in: {self.checkpoint_dir}")
        print("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SemRoCL Stage 2 Training (Optimized)')
    parser.add_argument('--config', type=str, default='train_stage2_curriculum.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {args.config}")
    
    # Create trainer
    trainer = Stage2Trainer(config)
    
    # Start training
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\nWarning: Training interrupted by user")
        print(f"  Saving checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, 0.0, is_best=False)
        print(f"Checkpoint saved, you can resume with --resume")
    except Exception as e:
        print(f"\nError: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
