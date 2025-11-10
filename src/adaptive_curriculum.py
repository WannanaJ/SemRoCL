"""
Adaptive Curriculum Learning Module
Dynamically adjusts loss weights based on validation metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum learning scheduler
    Adjusts loss weights dynamically based on training progress
    """
    
    def __init__(self, config: Dict, save_dir: Optional[str] = None):
        """
        Args:
            config: Training configuration
            save_dir: Directory to save curriculum history
        """
        self.config = config
        self.curriculum_config = config.get('curriculum', {})
        self.adaptive_enabled = self.curriculum_config.get('adaptive', False)
        
        # Save directory
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Base weights (target values)
        self.base_weights = {
            'w_color': config['loss'].get('w_color', 0.0002),
            'w_semantic': config['loss'].get('w_semantic', 0.6),
            'w_adv': config['loss'].get('w_adv', 0.05),
            'w_freq': config['loss'].get('w_freq', 0.00001),
            'w_task': config['loss'].get('w_task', 0.0),
            'w_perceptual': config['loss'].get('w_perceptual', 1.0)
        }
        
        # Current weights
        self.current_weights = self.base_weights.copy()
        
        # Adaptive parameters
        self.adaptive_params = {
            'psnr_threshold': 15.0,  # PSNR threshold for advancing
            'ssim_threshold': 0.45,  # SSIM threshold for advancing
            'patience': 5,  # Epochs to wait before adjusting
            'speed_factor': 1.2,  # Multiplier for fast progress
            'slow_factor': 0.8,  # Multiplier for slow progress
            'min_weight_scale': 0.5,  # Minimum weight scale
            'max_weight_scale': 1.5   # Maximum weight scale
        }
        
        # Update from config
        if 'adaptive_params' in self.curriculum_config:
            self.adaptive_params.update(self.curriculum_config['adaptive_params'])
        
        # Tracking
        self.history = {
            'epoch': [],
            'psnr': [],
            'ssim': [],
            'weights': [],
            'phase': [],
            'adjustments': []
        }
        
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.epochs_since_improvement = 0
        self.current_phase = 1
        
        print(f"✓ Adaptive Curriculum Scheduler initialized")
        print(f"  Adaptive mode: {self.adaptive_enabled}")
        if self.adaptive_enabled:
            print(f"  PSNR threshold: {self.adaptive_params['psnr_threshold']:.1f}")
            print(f"  SSIM threshold: {self.adaptive_params['ssim_threshold']:.2f}")
            print(f"  Patience: {self.adaptive_params['patience']} epochs")
    
    def get_phase(self, epoch: int) -> int:
        """Get current curriculum phase based on epoch"""
        phase_configs = {
            1: self.curriculum_config.get('phase1', {}),
            2: self.curriculum_config.get('phase2', {}),
            3: self.curriculum_config.get('phase3', {}),
            4: self.curriculum_config.get('phase4', {}),
            5: self.curriculum_config.get('phase5', {}),
            6: self.curriculum_config.get('phase6', {})
        }
        
        for phase, phase_cfg in phase_configs.items():
            epoch_range = phase_cfg.get('epochs', [1, 1])
            if epoch >= epoch_range[0] and epoch <= epoch_range[1]:
                return phase
        return 6
    
    def get_base_weights(self, epoch: int) -> Dict[str, float]:
        """Get base curriculum weights for current epoch (non-adaptive)"""
        if not self.curriculum_config.get('enabled', False):
            return self.base_weights.copy()
        
        phase = self.get_phase(epoch)
        phase_cfg = self.curriculum_config.get(f'phase{phase}', {})
        loss_weights = phase_cfg.get('loss_weights', {})
        
        # Interpolate weights
        epoch_range = phase_cfg.get('epochs', [1, 1])
        start_epoch, end_epoch = epoch_range[0], epoch_range[1]
        progress = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
        
        weights = {}
        for key, value in loss_weights.items():
            if isinstance(value, list) and len(value) == 2:
                # Linear interpolation
                weights[key] = value[0] + progress * (value[1] - value[0])
            else:
                weights[key] = value
        
        return weights
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update curriculum based on metrics
        
        Args:
            epoch: Current epoch
            metrics: Dictionary with 'psnr' and 'ssim' keys
        
        Returns:
            Updated loss weights
        """
        psnr = metrics.get('psnr', 0.0)
        ssim = metrics.get('ssim', 0.0)
        
        # Get base weights (from curriculum config)
        base_weights = self.get_base_weights(epoch)
        
        if not self.adaptive_enabled:
            # Non-adaptive: just use base weights
            self.current_weights = base_weights
        else:
            # Adaptive: adjust based on metrics
            adjustment = self._compute_adjustment(epoch, psnr, ssim)
            
            # Apply adjustment to base weights
            adjusted_weights = {}
            for key, base_val in base_weights.items():
                adjusted_weights[key] = base_val * adjustment
            
            self.current_weights = adjusted_weights
        
        # Update history
        self.current_phase = self.get_phase(epoch)
        self.history['epoch'].append(epoch)
        self.history['psnr'].append(psnr)
        self.history['ssim'].append(ssim)
        self.history['weights'].append(self.current_weights.copy())
        self.history['phase'].append(self.current_phase)
        
        # Save history
        if self.save_dir and epoch % 10 == 0:
            self.save_history()
        
        return self.current_weights.copy()
    
    def _compute_adjustment(self, epoch: int, psnr: float, ssim: float) -> float:
        """
        Compute adaptive adjustment factor
        
        Returns:
            Adjustment factor (multiplier for weights)
        """
        # Check if metrics improved
        psnr_improved = psnr > self.best_psnr
        ssim_improved = ssim > self.best_ssim
        
        if psnr_improved or ssim_improved:
            # Update best metrics
            if psnr_improved:
                self.best_psnr = psnr
            if ssim_improved:
                self.best_ssim = ssim
            
            self.epochs_since_improvement = 0
            
            # Fast progress: increase difficulty (increase semantic/perceptual)
            adjustment = self.adaptive_params['speed_factor']
            self.history['adjustments'].append(f"Epoch {epoch}: Speed up ({adjustment:.2f}x)")
            
        else:
            # No improvement
            self.epochs_since_improvement += 1
            
            if self.epochs_since_improvement >= self.adaptive_params['patience']:
                # Slow progress: reduce difficulty
                adjustment = self.adaptive_params['slow_factor']
                self.history['adjustments'].append(
                    f"Epoch {epoch}: Slow down ({adjustment:.2f}x) - No improvement for {self.epochs_since_improvement} epochs"
                )
                
                # Reset counter after adjustment
                self.epochs_since_improvement = 0
            else:
                # Keep current weights
                adjustment = 1.0
        
        # Clamp adjustment
        adjustment = np.clip(
            adjustment,
            self.adaptive_params['min_weight_scale'],
            self.adaptive_params['max_weight_scale']
        )
        
        return adjustment
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.current_weights.copy()
    
    def save_history(self):
        """Save curriculum history to file"""
        if not self.save_dir:
            return
        
        history_path = os.path.join(self.save_dir, 'curriculum_history.json')
        
        # Convert to JSON-serializable format
        history_json = {
            'epoch': self.history['epoch'],
            'psnr': self.history['psnr'],
            'ssim': self.history['ssim'],
            'phase': self.history['phase'],
            'adjustments': self.history['adjustments'],
            'weights': [
                {k: float(v) for k, v in w.items()}
                for w in self.history['weights']
            ]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"  ✓ Saved curriculum history to {history_path}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot curriculum history
        
        Args:
            save_path: Path to save plot (if None, just display)
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.history['epoch']) == 0:
                print("No history to plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: PSNR over time
            ax1 = axes[0, 0]
            ax1.plot(self.history['epoch'], self.history['psnr'], 'b-', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('PSNR (dB)')
            ax1.set_title('PSNR Progress')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: SSIM over time
            ax2 = axes[0, 1]
            ax2.plot(self.history['epoch'], self.history['ssim'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('SSIM')
            ax2.set_title('SSIM Progress')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Loss weights over time
            ax3 = axes[1, 0]
            for key in ['w_semantic', 'w_perceptual', 'w_color']:
                weights = [w.get(key, 0) for w in self.history['weights']]
                ax3.plot(self.history['epoch'], weights, label=key, linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Weight')
            ax3.set_title('Loss Weights Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Phase progression
            ax4 = axes[1, 1]
            ax4.plot(self.history['epoch'], self.history['phase'], 'r-', linewidth=2, marker='o')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Curriculum Phase')
            ax4.set_title('Curriculum Phase Progression')
            ax4.set_yticks(range(1, 7))
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Saved curriculum plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠ matplotlib not installed, skipping plot")


class SmartCurriculumScheduler(AdaptiveCurriculumScheduler):
    """
    Enhanced adaptive scheduler with more intelligent strategies
    """
    
    def __init__(self, config: Dict, save_dir: Optional[str] = None):
        super().__init__(config, save_dir)
        
        # Additional tracking for smart decisions
        self.loss_history = []
        self.stagnation_threshold = 3  # Consecutive epochs without improvement
        self.phase_skip_enabled = True  # Can skip phases if progressing well
    
    def _compute_adjustment(self, epoch: int, psnr: float, ssim: float) -> float:
        """
        Enhanced adjustment with smarter strategies
        """
        # Record current metrics
        current_score = psnr + ssim * 100  # Combined score
        
        # Check trend
        if len(self.loss_history) >= 3:
            recent_scores = self.loss_history[-3:]
            trend = (current_score - recent_scores[0]) / len(recent_scores)
            
            if trend > 1.0:
                # Strong positive trend: aggressive increase
                adjustment = 1.3
                self.history['adjustments'].append(
                    f"Epoch {epoch}: Strong progress - aggressive increase"
                )
            elif trend > 0.2:
                # Moderate progress: normal increase
                adjustment = 1.1
            elif trend < -0.5:
                # Negative trend: reduce difficulty
                adjustment = 0.7
                self.history['adjustments'].append(
                    f"Epoch {epoch}: Regression - reduce difficulty"
                )
            else:
                # Stable or slight progress
                adjustment = 1.0
        else:
            # Not enough history
            adjustment = 1.0
        
        self.loss_history.append(current_score)
        
        # Keep only recent history
        if len(self.loss_history) > 20:
            self.loss_history = self.loss_history[-20:]
        
        # Clamp
        adjustment = np.clip(adjustment, 0.5, 1.5)
        
        return adjustment


if __name__ == '__main__':
    # Test adaptive curriculum
    print("="*60)
    print("Testing Adaptive Curriculum Scheduler")
    print("="*60)
    
    # Mock config
    config = {
        'loss': {
            'w_color': 0.0002,
            'w_semantic': 0.6,
            'w_perceptual': 1.0,
            'w_adv': 0.05,
            'w_freq': 0.00001,
            'w_task': 0.0
        },
        'curriculum': {
            'enabled': True,
            'adaptive': True,
            'adaptive_params': {
                'psnr_threshold': 15.0,
                'ssim_threshold': 0.45,
                'patience': 3
            },
            'phase1': {'epochs': [1, 20], 'loss_weights': {'w_semantic': 0.0, 'w_perceptual': 0.01}},
            'phase2': {'epochs': [21, 50], 'loss_weights': {'w_semantic': [0.0, 0.2], 'w_perceptual': 0.2}},
            'phase3': {'epochs': [51, 100], 'loss_weights': {'w_semantic': 0.4, 'w_perceptual': 0.6}}
        }
    }
    
    # Create scheduler
    scheduler = AdaptiveCurriculumScheduler(config, save_dir='./test_curriculum')
    
    # Simulate training
    print("\nSimulating training progress:")
    print("-" * 60)
    
    # Simulate improving metrics
    for epoch in range(1, 101, 5):
        # Simulate PSNR/SSIM improvement with noise
        psnr = 10 + epoch * 0.08 + np.random.randn() * 0.5
        ssim = 0.2 + epoch * 0.003 + np.random.randn() * 0.02
        
        metrics = {'psnr': psnr, 'ssim': ssim}
        weights = scheduler.update(epoch, metrics)
        
        print(f"Epoch {epoch:3d} | Phase {scheduler.current_phase} | "
              f"PSNR: {psnr:5.2f} | SSIM: {ssim:.3f} | "
              f"w_semantic: {weights['w_semantic']:.3f}")
    
    # Plot history
    print("\n" + "-" * 60)
    scheduler.plot_history('./test_curriculum/curriculum_plot.png')
    
    # Print adjustments
    print("\nAdjustments made:")
    for adj in scheduler.history['adjustments'][-5:]:
        print(f"  {adj}")
    
    print("\n✓ Test completed!")
