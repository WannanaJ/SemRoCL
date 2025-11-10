"""
Enhanced Stage 2 Training Script with Advanced Features
- Multi-scale enhancement
- Full semantic guidance
- Adaptive curriculum learning

Key improvements:
- Fixed missing F import
- Better error handling for imports
- Removed emoji characters
- Improved gradient management
"""

import os
import sys
import yaml
import argparse
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings("ignore")

from metrics_utils import (
    compute_psnr,
    compute_ssim,
    compute_iter_speed,
    compute_delta_e,
    maybe_compute_niqe,
    maybe_compute_brisque,
)

try:
    import lpips  # type: ignore
    _LPIPS_AVAILABLE = True
except ImportError:
    lpips = None  # type: ignore
    _LPIPS_AVAILABLE = False
# Import modules with error handling
try:
    from data_loader import get_dataloader
except ImportError as e:
    raise ImportError(f"Cannot import data_loader: {e}")

try:
    from loss_functions import CombinedLoss
except ImportError as e:
    raise ImportError(f"Cannot import loss_functions: {e}")

try:
    # 浼樺厛鍔犺浇澧炲己鐗堟ā鍨嬶紙Claude 鐨勶級
    from model.models_enhanced import (
        EnhancementGenerator as EnhancedGenerator,
        SemanticGuidanceModule,
        Discriminator as EnhancedDiscriminator
    )
    EnhancementGenerator = EnhancedGenerator
    Discriminator = EnhancedDiscriminator
    print(" Loaded enhanced generator & discriminator (models_enhanced.py)")
except ImportError:
    # Fallback to base models
    from model.models import EnhancementGenerator, Discriminator
    SemanticGuidanceModule = None  # type: ignore[assignment]
    print(" Using base generator & discriminator (models.py) - semantic module unavailable")


try:
    from adaptive_curriculum import AdaptiveCurriculumScheduler, SmartCurriculumScheduler
except ImportError as e:
    raise ImportError(f"Cannot import adaptive_curriculum: {e}. Ensure adaptive_curriculum.py exists.")


class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        

        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EnhancedStage2Trainer:
    """
    Enhanced trainer with multi-scale, semantic guidance, and adaptive curriculum
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_requires_grad = False
        self.semantic_proj: Optional[nn.Module] = None
        self.monitor_cfg = self.config.get('monitoring', {})
        self.metrics_interval = int(self.monitor_cfg.get('metrics_interval', 50))
        self.semantic_conf_threshold = float(self.config['model'].get('semantic_conf_threshold', 0.0))
        
        # Feature flags
        self.use_multiscale = config['model'].get('use_multiscale', False)
        self.use_semantic_guidance = config['model'].get('use_semantic_guidance', True)
        self.use_adaptive_curriculum = config['curriculum'].get('adaptive', False)
        
        print(f"\n{'='*70}")
        print(f"Enhanced Stage 2 Trainer")
        print(f"{'='*70}")
        print(f"Multi-scale enhancement: {self.use_multiscale}")
        print(f"Semantic guidance: {self.use_semantic_guidance}")
        print(f"Adaptive curriculum: {self.use_adaptive_curriculum}")
        print(f"{'='*70}\n")
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize loss function
        self.criterion = CombinedLoss(config).to(self.device)
        
        # Adaptive curriculum scheduler
        if self.use_adaptive_curriculum:
            SchedulerClass = SmartCurriculumScheduler if config['curriculum'].get('smart', False) else AdaptiveCurriculumScheduler
            self.curriculum_scheduler = SchedulerClass(
                config,
                save_dir=os.path.join(config['output_dir'], config['exp_name'], 'curriculum')
            )
        else:
            self.curriculum_scheduler = None
        
        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler_g = GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.scaler_d = GradScaler(enabled=self.use_amp) if self.use_amp else None
        if self.use_amp and (self.scaler_g is None or self.scaler_d is None):
            raise RuntimeError("AMP is enabled but GradScaler could not be initialized.")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.current_phase = 1

        
        # Setup logging
        self.setup_logging()

        # LPIPS for in-training monitoring
        self.lpips_model = None
        if _LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
                print("LPIPS model loaded for training metrics.")
            except Exception as exc:
                print(f"Warning: failed to load LPIPS model ({exc})")
        
        # Gradient accumulation
        self.accumulate_steps = config['training'].get('accumulate_steps', 1)
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)
        self.gradient_clip_d = config['training'].get('gradient_clip_discriminator', 5.0)
        
        # Discriminator schedule
        self.disc_start_epoch = config['training'].get('disc_start_epoch', 90)
        self.disc_update_freq = config['training'].get('disc_update_freq', 5)
        
        # EMA
        if config['training'].get('use_ema', False):
            self.ema = EMA(self.generator, decay=config['training'].get('ema_decay', 0.999))
        else:
            self.ema = None
        
        print(f"Enhanced Trainer initialized on {self.device}")
    
    def setup_models(self):
        """Initialize all models"""
        model_cfg = self.config['model']
        
        # Generator configuration
        gen_config = {
            'enhancer_channels': model_cfg.get('enhancer_channels', 32),
            'num_iterations': model_cfg.get('num_iterations', 8),
            'semantic_channels': model_cfg.get('semantic_channels', 64),
            'use_semantic': model_cfg.get('use_semantic', True),
            'use_multiscale': self.use_multiscale,
            'num_scales': model_cfg.get('num_scales', 3),
            'lightweight': model_cfg.get('lightweight', False),
            'channel_multiplier': model_cfg.get('channel_multiplier', 1.0),
            'use_depthwise': model_cfg.get('use_depthwise', model_cfg.get('lightweight', False)),
            'use_noise_gate': model_cfg.get('use_noise_gate', True),
            'noise_gate_kernel': model_cfg.get('noise_gate_kernel', 3),
        }
        
        self.generator = EnhancementGenerator(gen_config).to(self.device)
        
        # Semantic guidance module
        if self.use_semantic_guidance:
            if SemanticGuidanceModule is None:
                print("Semantic module unavailable; disabling semantic guidance.")
                self.use_semantic_guidance = False
                self.semantic_module = None
            else:
                semantic_feature_dim = model_cfg.get('semantic_feature_dim', model_cfg.get('semantic_channels', 64))
                downsample_ratio = model_cfg.get('semantic_downsample', 1)
                self.semantic_module = SemanticGuidanceModule(
                    num_classes=model_cfg.get('num_classes', 19),
                    pretrained=model_cfg.get('semantic_pretrained', True),
                    output_features=True,
                    downsample_ratio=downsample_ratio,
                    feature_dim=semantic_feature_dim
                ).to(self.device)

                proj_in = getattr(self.semantic_module, 'output_channels', semantic_feature_dim)
                proj_out = model_cfg.get('semantic_channels', 64)
                if proj_in != proj_out:
                    self.semantic_proj = nn.Conv2d(proj_in, proj_out, 1).to(self.device)
                else:
                    self.semantic_proj = nn.Identity().to(self.device)

                self.semantic_requires_grad = bool(model_cfg.get('finetune_semantic', False))
                if not self.semantic_requires_grad:
                    for param in self.semantic_module.parameters():
                        param.requires_grad = False
                    self.semantic_module.eval()
                else:
                    self.semantic_module.train()
                print(
                    f"Semantic guidance: feature dim {proj_in}->{proj_out}, "
                    f"downsample ratio x{downsample_ratio}"
                )
        else:
            self.semantic_module = None
            self.semantic_proj = None
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
        print(f"Generator: {gen_params:,} parameters")
        print(f"Discriminator: {disc_params:,} parameters")
        
        if self.semantic_module:
            sem_params = sum(p.numel() for p in self.semantic_module.parameters())
            if self.semantic_proj is not None:
                sem_params += sum(p.numel() for p in self.semantic_proj.parameters())
            print(f"Semantic Module: {sem_params:,} parameters")
    
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
        
        # Semantic module optimizer (if fine-tuning)
        if self.semantic_module and self.semantic_requires_grad:
            semantic_params = list(self.semantic_module.parameters())
            if self.semantic_proj is not None:
                semantic_params += [p for p in self.semantic_proj.parameters() if p.requires_grad]
            self.optimizer_sem = optim.AdamW(
                semantic_params,
                lr=train_cfg.get('lr_semantic', 0.00002),
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        else:
            self.optimizer_sem = None
        
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
            
            if self.optimizer_sem:
                self.scheduler_sem = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_sem,
                    T_max=train_cfg['epochs'],
                    eta_min=1e-8
                )
        else:
            milestones = [60, 120, 170]
            self.scheduler_g = optim.lr_scheduler.MultiStepLR(
                self.optimizer_g, milestones=milestones, gamma=0.5
            )
            self.scheduler_d = optim.lr_scheduler.MultiStepLR(
                self.optimizer_d, milestones=milestones, gamma=0.5
            )
            if self.optimizer_sem:
                self.scheduler_sem = optim.lr_scheduler.MultiStepLR(
                    self.optimizer_sem, milestones=milestones, gamma=0.5
                )
        
        print("Optimizers initialized")
    
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
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.csv_path = os.path.join(self.log_dir, 'training_log.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'gen_loss', 'disc_loss',
                'color_loss', 'semantic_loss', 'perceptual_loss', 
                'adv_loss', 'freq_loss', 'noise_loss', 'lr', 'w_semantic'
            ])
        
        self.metrics_path = os.path.join(self.log_dir, 'metrics.csv')
        with open(self.metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'step', 'phase', 'psnr', 'ssim', 'delta_e', 'niqe', 'brisque', 'lpips',
                'w_color', 'w_semantic', 'w_perceptual', 'w_freq', 'w_adv',
                'lr', 'iter_time_ms', 'speed_imgs_sec'
            ])
        
        print(f"Logging initialized at {self.output_dir}")
    
    def get_current_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get loss weights for current epoch based on curriculum"""
        curriculum_cfg = self.config.get('curriculum', {})
        
        if not curriculum_cfg.get('enabled', False):
            return self.config['loss']
        
        for phase_num in range(1, 7):
            phase_key = f'phase{phase_num}'
            if phase_key not in curriculum_cfg:
                continue
            
            phase = curriculum_cfg[phase_key]
            epoch_range = phase['epochs']
            
            if epoch_range[0] <= epoch <= epoch_range[1]:
                self.current_phase = phase_num
                weights = phase['loss_weights'].copy()
                
                for key, value in weights.items():
                    if isinstance(value, list) and len(value) == 2:
                        start_val, end_val = value
                        progress = (epoch - epoch_range[0]) / (epoch_range[1] - epoch_range[0])
                        weights[key] = start_val + (end_val - start_val) * progress
                
                return weights
        
        return self.config['loss']
    
    def extract_semantic_features(
        self,
        image: torch.Tensor,
        paths: Optional[List[str]] = None
    ):
        """Extract semantic features from image (paths reserved for future caching)."""
        if self.semantic_module is None:
            return None, None

        context = torch.enable_grad() if self.semantic_requires_grad else torch.no_grad()
        with context:
            output = self.semantic_module(image)

        if output is None:
            return None, None

        features = None
        confidence = None

        if isinstance(output, (list, tuple)):
            if len(output) == 3:
                _, confidence, features = output
            elif len(output) == 2:
                features, confidence = output
            elif len(output) == 1:
                features = output[0]
            else:
                raise RuntimeError(f"Unexpected semantic output tuple length: {len(output)}")
        elif isinstance(output, dict):
            features = output.get("features") or output.get("seg_features")
            confidence = output.get("confidence")
        else:
            features = output

        if features is None:
            print("Warning: Semantic module returned no features")
            return None, confidence

        if self.semantic_proj is not None:
            features = self.semantic_proj(features)

        if confidence is not None and self.semantic_conf_threshold > 0:
            mask = (confidence >= self.semantic_conf_threshold).float()
            confidence = confidence * mask
            features = features * mask

        expected_channels = self.config['model'].get('semantic_channels', 64)
        if features.shape[1] != expected_channels:
            print(f"Warning: Expected {expected_channels} semantic channels, got {features.shape[1]}")

        return features, confidence

    
    def clip_gradients_safely(self, model, max_norm: float) -> float:
        """Clip gradients safely and return norm"""
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        return total_norm.item()
    
    def train_epoch(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        current_weights = self.get_current_loss_weights(epoch)
        self.criterion.update_weights(current_weights)
        
        epoch_losses = {
            'gen_total': 0.0, 'disc_total': 0.0,
            'color': 0.0, 'semantic': 0.0, 'perceptual': 0.0,
            'adv': 0.0, 'freq': 0.0, 'noise': 0.0
        }
        
        num_batches = 0
        current_lr = self.optimizer_g.param_groups[0]['lr']
        current_w_semantic = current_weights.get('w_semantic', 0.0)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config['training']['epochs']} [Phase {self.current_phase}]")
        
        for batch_idx, batch in enumerate(pbar):
            iter_start = time.time()
            low_img = batch['low'].to(self.device)
            high_img = batch['high'].to(self.device)
            
            # Input validation
            if torch.isnan(low_img).any() or torch.isnan(high_img).any():
                print(f"Warning: NaN in input at batch {batch_idx}, skipping...")
                continue
            
            # Extract semantic features
            semantic_features, confidence_map = self.extract_semantic_features(low_img)
            
            # Train Generator
            should_train_g = (batch_idx % self.accumulate_steps == 0) or (batch_idx == len(dataloader) - 1)
            
            if should_train_g:
                self.optimizer_g.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Generate enhanced image
                if self.use_semantic_guidance and semantic_features is not None:
                    enhanced = self.generator(low_img, semantic_features, confidence_map)
                else:
                    enhanced = self.generator(low_img)
                
                # Validate output
                if torch.isnan(enhanced).any():
                    print(f"Warning: NaN in generator output at batch {batch_idx}, skipping...")
                    continue
                
                # Compute losses
                losses_g = self.criterion(
                    enhanced,
                    high_img,
                    low_img,
                    epoch=epoch
                )
                
                if torch.isnan(losses_g['total']):
                    print(f"Warning: NaN in loss at batch {batch_idx}, skipping...")
                    continue
                
                # Adversarial loss
                if epoch >= self.disc_start_epoch and batch_idx % self.disc_update_freq == 0:
                    fake_pred = self.discriminator(enhanced)
                    adv_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, 
                        torch.ones_like(fake_pred) * 0.9
                    )
                    losses_g['adv'] = adv_loss
                    losses_g['total'] = losses_g['total'] + current_weights.get('w_adv', 0.0) * adv_loss
            
            # Backward
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
                    self.clip_gradients_safely(self.generator, self.gradient_clip)
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                else:
                    self.clip_gradients_safely(self.generator, self.gradient_clip)
                    self.optimizer_g.step()
                
                if self.ema:
                    self.ema.update()
            
            # Train Discriminator
            disc_loss = torch.tensor(0.0).to(self.device)
            if epoch >= self.disc_start_epoch and batch_idx % self.disc_update_freq == 0:
                self.optimizer_d.zero_grad()
                
                with autocast(enabled=self.use_amp):
                    real_pred = self.discriminator(high_img)
                    real_loss = F.binary_cross_entropy_with_logits(
                        real_pred, 
                        torch.ones_like(real_pred) * 0.9
                    )
                    
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
                    self.clip_gradients_safely(self.discriminator, self.gradient_clip_d)
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    disc_loss.backward()
                    self.clip_gradients_safely(self.discriminator, self.gradient_clip_d)
                    self.optimizer_d.step()
            
            # Accumulate losses
            epoch_losses['gen_total'] += losses_g['total'].item()
            epoch_losses['disc_total'] += disc_loss.item()
            for key in ['color', 'semantic', 'perceptual', 'adv', 'freq', 'noise']:
                if key in losses_g:
                    epoch_losses[key] += losses_g[key].item()
            
            num_batches += 1
            iter_duration = time.time() - iter_start
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{losses_g['total'].item():.4f}",
                'D': f"{disc_loss.item():.4f}",
                'LR': f"{current_lr:.6f}",
                'Phase': self.current_phase
            })
            
            # Logging
            if self.global_step % 50 == 0:
                for key in ['total', 'color', 'semantic', 'perceptual', 'adv', 'freq', 'noise']:
                    if key in losses_g:
                        self.writer.add_scalar(f'Loss/{key}', losses_g[key].item(), self.global_step)
            
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
                        losses_g.get('noise', torch.tensor(0)).item(),
                        current_lr, current_w_semantic
                    ])

            if self.global_step % self.metrics_interval == 0:
                with torch.no_grad():
                    enhanced_detached = enhanced.detach()
                    step_psnr = compute_psnr(enhanced_detached, high_img).item()
                    step_ssim = compute_ssim(enhanced_detached, high_img).item()
                    step_delta_e = compute_delta_e(enhanced_detached, high_img).item()
                    step_niqe = maybe_compute_niqe(enhanced_detached)
                    step_brisque = maybe_compute_brisque(enhanced_detached)
                    step_lpips = None
                    if self.lpips_model is not None:
                        try:
                            enh_norm = enhanced_detached * 2.0 - 1.0
                            target_norm = high_img * 2.0 - 1.0
                            lpips_val = self.lpips_model(enh_norm, target_norm)
                            step_lpips = float(lpips_val.mean().item())
                        except Exception as exc:
                            self.writer.add_text('Warnings/LPIPS', f'LPIPS failed: {exc}', self.global_step)
                            step_lpips = None
                iter_ms, imgs_per_s = compute_iter_speed(iter_duration + 1e-8, low_img.size(0))
                logged_weights = current_weights
                if hasattr(self.criterion, 'get_last_weights'):
                    try:
                        logged_weights = self.criterion.get_last_weights()
                    except Exception:
                        logged_weights = current_weights
                w_color = logged_weights.get('w_color', 0.0)
                w_sem = logged_weights.get('w_semantic', 0.0)
                w_perc = logged_weights.get('w_perceptual', 0.0)
                w_freq = logged_weights.get('w_freq', 0.0)
                w_adv = logged_weights.get('w_adv', 0.0)
                self.writer.add_scalar('Train/PSNR', step_psnr, self.global_step)
                self.writer.add_scalar('Train/SSIM', step_ssim, self.global_step)
                self.writer.add_scalar('Train/DeltaE', step_delta_e, self.global_step)
                if step_niqe is not None:
                    self.writer.add_scalar('Train/NIQE', float(step_niqe), self.global_step)
                if step_brisque is not None:
                    self.writer.add_scalar('Train/BRISQUE', float(step_brisque), self.global_step)
                if step_lpips is not None:
                    self.writer.add_scalar('Train/LPIPS', float(step_lpips), self.global_step)
                with open(self.metrics_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, self.global_step, self.current_phase,
                        step_psnr, step_ssim, step_delta_e,
                        float(step_niqe) if step_niqe is not None else '',
                        float(step_brisque) if step_brisque is not None else '',
                        float(step_lpips) if step_lpips is not None else '',
                        w_color, w_sem, w_perc, w_freq, w_adv,
                        current_lr, iter_ms, imgs_per_s
                    ])
            
            if self.global_step % 1000 == 0:
                self.save_sample_images(low_img, enhanced, high_img, epoch)
            
            self.global_step += 1
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, epoch: int, dataloader: DataLoader) -> Dict[str, float]:
        """Validation with semantic features"""
        self.generator.eval()
        
        if self.ema:
            self.ema.apply_shadow()
        
        val_losses = {'total': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'delta_e': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                low_img = batch['low'].to(self.device)
                high_img = batch.get('high')
                
                if high_img is None:
                    continue
                
                high_img = high_img.to(self.device)
                
                # Extract semantic features
                semantic_features, confidence_map = self.extract_semantic_features(low_img)
                
                # Generate enhanced image
                if self.use_semantic_guidance and semantic_features is not None:
                    enhanced = self.generator(low_img, semantic_features, confidence_map)
                else:
                    enhanced = self.generator(low_img)
                
                # Compute metrics
                mse = F.mse_loss(enhanced, high_img)
                val_losses['total'] += mse.item()
                
                psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
                val_losses['psnr'] += psnr.item()
                val_losses['delta_e'] += compute_delta_e(enhanced, high_img).item()
                
                if hasattr(self.criterion, 'ssim_loss'):
                    ssim_val = self.criterion.ssim_loss(enhanced, high_img)
                    val_losses['ssim'] += (1 - ssim_val.item())
                
                num_batches += 1
        
        if self.ema:
            self.ema.restore()
        
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        # Log metrics
        self.writer.add_scalar('Val/Loss', val_losses['total'], epoch)
        self.writer.add_scalar('Val/PSNR', val_losses['psnr'], epoch)
        self.writer.add_scalar('Val/SSIM', val_losses['ssim'], epoch)
        self.writer.add_scalar('Val/DeltaE', val_losses['delta_e'], epoch)
        
        self.generator.train()
        
        return val_losses
    
    def save_sample_images(self, low, enhanced, target, epoch):
        """
        Save comparison images (low/enhanced/high) while avoiding duplicate files.
        """
        import torchvision.utils as vutils

        with torch.no_grad():
            enhanced_processed = torch.clamp(enhanced.clone(), 0, 1)
            kernel = self._get_gaussian_kernel(3, 0.5).to(enhanced.device)
            for i in range(enhanced_processed.size(1)):
                enhanced_processed[:, i:i+1] = F.conv2d(
                    enhanced_processed[:, i:i+1],
                    kernel,
                    padding=1
                )

        low_img = low[0:1]
        enhanced_img = enhanced_processed[0:1]
        target_img = target[0:1]
        comparison = torch.cat([low_img, enhanced_img, target_img], dim=3)

        save_path = os.path.join(self.image_dir, f'epoch_{epoch:03d}_step_{self.global_step}.png')
        vutils.save_image(comparison, save_path, normalize=False, format='PNG')

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
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        if self.ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if epoch % self.config['training'].get('save_freq', 5) == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        print("\n" + "="*70)
        print(" Enhanced Stage 2 Training")
        print("="*70)
        print(f" Total epochs: {self.config['training']['epochs']}")
        print(f" Multi-scale: {self.use_multiscale}")
        print(f" Semantic guidance: {self.use_semantic_guidance}")
        print(f" Adaptive curriculum: {self.use_adaptive_curriculum}")
        print("="*70 + "\n")
        
        # Create dataloaders
        train_loader = get_dataloader(self.config, split='train')
        
        val_dir = os.path.join(self.config['dataset']['root_dir'], 'val')
        val_loader = get_dataloader(self.config, split='val') if os.path.exists(val_dir) else None
        
        # Training loop
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch(epoch, train_loader)
            
            # Validate
            val_losses = {'total': 0, 'psnr': 0, 'ssim': 0}
            if val_loader and epoch % 5 == 0:
                val_losses = self.validate(epoch, val_loader)
                
                # Update adaptive curriculum
                if self.curriculum_scheduler:
                    weights = self.curriculum_scheduler.update(epoch, val_losses)
                    self.criterion.update_weights(weights)
            
            # Update LR
            self.scheduler_g.step()
            self.scheduler_d.step()
            if self.optimizer_sem:
                self.scheduler_sem.step()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f" Epoch {epoch} Summary")
            print(f"{'='*70}")
            print(f"  G Loss: {train_losses['gen_total']:.4f}")
            print(f"  D Loss: {train_losses['disc_total']:.4f}")
            if val_loader and epoch % 5 == 0:
                print(f"  Val PSNR: {val_losses['psnr']:.2f} dB")
                print(f"  Val SSIM: {val_losses['ssim']:.4f}")
            print("=" * 70 + "\n")

            
            # Save
            is_best = val_losses['total'] < self.best_metric if val_loader else False
            if is_best:
                self.best_metric = val_losses['total']
            
            self.save_checkpoint(epoch, val_losses['total'], is_best)
        
        # Save curriculum history
        if self.curriculum_scheduler:
            self.curriculum_scheduler.save_history()
            self.curriculum_scheduler.plot_history(
                os.path.join(self.output_dir, 'curriculum_plot.png')
            )
        
        self.writer.close()
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Stage 2 Training')
    parser.add_argument('--config', type=str, default='train_stage2_enhanced.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = EnhancedStage2Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
