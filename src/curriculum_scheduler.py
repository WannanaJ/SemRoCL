"""
Curriculum Learning Scheduler for Stage 2 Training
Manages smooth phase transitions and dynamic weight adjustments
"""

import numpy as np
from typing import Dict, Tuple, Any


class CurriculumScheduler:
    """
    Manages curriculum learning phases with smooth transitions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum scheduler
        
        Args:
            config: Curriculum configuration from YAML
        """
        self.enabled = config.get('enabled', True)
        self.current_phase = 0
        self.phases = []
        
        if not self.enabled:
            # Default weights if curriculum is disabled
            self.default_weights = {
                'w_color': 0.0001,
                'w_semantic': 1.0,
                'w_adv': 0.5,
                'w_freq': 0.00001,
                'w_task': 0.5,
                'w_perceptual': 2.0
            }
            return
        
        # Parse phases from config
        self._parse_phases(config)
        
        # Transition settings
        self.transition_grace_epochs = config.get('monitoring', {}).get('safety', {}).get('transition_grace_epochs', 5)
        self.enable_grace_period = config.get('monitoring', {}).get('safety', {}).get('enable_grace_period', True)
        
        # Tracking
        self.last_transition_epoch = -1
        self.phase_history = []
        
    def _parse_phases(self, config: Dict[str, Any]):
        """Parse phase configurations"""
        # Collect all phase keys (phase1, phase2, etc.)
        phase_keys = sorted([k for k in config.keys() if k.startswith('phase')])
        
        for phase_key in phase_keys:
            phase_config = config[phase_key]
            phase = {
                'name': phase_key,
                'epochs': phase_config['epochs'],
                'description': phase_config.get('description', ''),
                'loss_weights': phase_config['loss_weights'],
                'monitoring': phase_config.get('monitoring', {})
            }
            self.phases.append(phase)
    
    def get_current_phase(self, epoch: int) -> int:
        """Get the current phase index based on epoch"""
        for i, phase in enumerate(self.phases):
            start_epoch, end_epoch = phase['epochs']
            if start_epoch <= epoch + 1 <= end_epoch:  # epochs are 1-indexed in config
                return i
        return len(self.phases) - 1  # Return last phase if beyond all phases
    
    def get_phase_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for current epoch with smooth transitions
        
        Args:
            epoch: Current training epoch (0-indexed)
        
        Returns:
            Dictionary of loss weights
        """
        if not self.enabled:
            return self.default_weights
        
        phase_idx = self.get_current_phase(epoch)
        
        if phase_idx >= len(self.phases):
            phase_idx = len(self.phases) - 1
        
        phase = self.phases[phase_idx]
        weights = {}
        
        # Get epoch range for this phase
        start_epoch, end_epoch = phase['epochs']
        epoch_in_phase = epoch + 1 - start_epoch  # Convert to 1-indexed
        phase_duration = end_epoch - start_epoch + 1
        
        # Process each weight
        for key, value in phase['loss_weights'].items():
            if isinstance(value, list) and len(value) == 2:
                # Linear interpolation between start and end values
                start_val, end_val = value
                progress = min(1.0, epoch_in_phase / max(1, phase_duration))
                
                # Smooth transition using cosine annealing
                if self._use_smooth_transition(key):
                    progress = 0.5 * (1 - np.cos(np.pi * progress))
                
                weights[key] = start_val + (end_val - start_val) * progress
            else:
                # Static value
                weights[key] = value
        
        return weights
    
    def _use_smooth_transition(self, weight_key: str) -> bool:
        """Determine if smooth transition should be used for this weight"""
        # Use smooth transitions for critical weights
        smooth_weights = ['w_semantic', 'w_adv', 'w_perceptual']
        return weight_key in smooth_weights
    
    def check_phase_transition(self, epoch: int) -> Dict[str, Any]:
        """
        Check if a phase transition occurred
        
        Args:
            epoch: Current epoch
        
        Returns:
            Dictionary with transition information
        """
        current_phase = self.get_current_phase(epoch)
        
        # Check if we transitioned
        transitioned = False
        if len(self.phase_history) > 0 and current_phase != self.phase_history[-1]:
            transitioned = True
            self.last_transition_epoch = epoch
        
        self.phase_history.append(current_phase)
        
        if not self.enabled:
            return {
                'transitioned': False,
                'current_phase': 0,
                'description': 'Curriculum disabled',
                'weights': self.default_weights
            }
        
        phase_info = self.phases[current_phase] if current_phase < len(self.phases) else self.phases[-1]
        
        return {
            'transitioned': transitioned,
            'current_phase': current_phase,
            'phase_name': phase_info['name'],
            'description': phase_info['description'],
            'weights': self.get_phase_weights(epoch),
            'in_grace_period': self.is_in_grace_period(epoch)
        }
    
    def is_in_grace_period(self, epoch: int) -> bool:
        """Check if we're in a grace period after phase transition"""
        if not self.enable_grace_period:
            return False
        
        if self.last_transition_epoch < 0:
            return False
        
        return (epoch - self.last_transition_epoch) < self.transition_grace_epochs
    
    def get_monitoring_targets(self, epoch: int) -> Dict[str, float]:
        """
        Get monitoring targets for current phase
        
        Returns:
            Dictionary of target metrics
        """
        if not self.enabled:
            return {}
        
        phase_idx = self.get_current_phase(epoch)
        if phase_idx >= len(self.phases):
            phase_idx = len(self.phases) - 1
        
        phase = self.phases[phase_idx]
        return phase.get('monitoring', {})
    
    def should_skip_safety_check(self, epoch: int) -> bool:
        """
        Determine if safety checks should be skipped (e.g., during grace period)
        
        Args:
            epoch: Current epoch
        
        Returns:
            True if safety checks should be skipped
        """
        return self.is_in_grace_period(epoch)
    
    def adjust_weights_on_instability(self, current_weights: Dict[str, float], 
                                     metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights if training becomes unstable
        
        Args:
            current_weights: Current loss weights
            metrics: Current training metrics
        
        Returns:
            Adjusted weights
        """
        adjusted_weights = current_weights.copy()
        
        # Check for NaN or extreme values
        if any(np.isnan(v) or np.isinf(v) for v in metrics.values() if isinstance(v, (int, float))):
            print(" NaN/Inf detected! Reducing loss weights...")
            # Reduce all weights by 50%
            for key in adjusted_weights:
                adjusted_weights[key] *= 0.5
        
        # Check for exploding discriminator loss
        if 'd_loss' in metrics and metrics['d_loss'] > 10.0:
            print(" Discriminator loss too high! Reducing adversarial weight...")
            adjusted_weights['w_adv'] *= 0.5
        
        # Check for collapsed enhancement (too dark/bright)
        if 'mean_enhanced' in metrics:
            mean_val = metrics['mean_enhanced']
            if mean_val < 0.1 or mean_val > 0.9:
                print(f" Enhancement collapsed (mean={mean_val:.3f})! Adjusting color weight...")
                adjusted_weights['w_color'] *= 2.0
        
        return adjusted_weights


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum that adjusts based on training progress
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Adaptive settings
        self.adaptive_enabled = config.get('adaptive', {}).get('enabled', False)
        self.min_phase_epochs = config.get('adaptive', {}).get('min_phase_epochs', 10)
        self.target_improvement = config.get('adaptive', {}).get('target_improvement', 0.01)
        
        # Performance tracking
        self.phase_start_metrics = {}
        self.phase_best_metrics = {}
        
    def should_advance_phase(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Determine if we should advance to the next phase based on performance
        
        Args:
            epoch: Current epoch
            metrics: Current performance metrics (PSNR, SSIM, losses)
        
        Returns:
            True if phase should advance
        """
        if not self.adaptive_enabled:
            return False  # Use fixed schedule
        
        current_phase = self.get_current_phase(epoch)
        phase_info = self.phases[current_phase]
        
        # Check minimum epochs in phase
        start_epoch = phase_info['epochs'][0]
        if (epoch + 1 - start_epoch) < self.min_phase_epochs:
            return False
        
        # Check performance improvement
        if current_phase not in self.phase_start_metrics:
            self.phase_start_metrics[current_phase] = metrics.copy()
            self.phase_best_metrics[current_phase] = metrics.copy()
            return False
        
        # Update best metrics
        if 'psnr' in metrics:
            if metrics['psnr'] > self.phase_best_metrics[current_phase].get('psnr', 0):
                self.phase_best_metrics[current_phase]['psnr'] = metrics['psnr']
        
        # Check if we've achieved target improvement
        start_psnr = self.phase_start_metrics[current_phase].get('psnr', 0)
        best_psnr = self.phase_best_metrics[current_phase].get('psnr', 0)
        
        improvement = best_psnr - start_psnr
        
        if improvement >= self.target_improvement:
            print(f" Phase {current_phase} target achieved! (Improvement: {improvement:.3f} dB)")
            return True
        
        return False
    
    def force_advance_phase(self, epoch: int):
        """Manually advance to next phase"""
        current_phase = self.get_current_phase(epoch)
        if current_phase < len(self.phases) - 1:
            next_phase = self.phases[current_phase + 1]
            # Adjust epoch ranges
            self.phases[current_phase]['epochs'] = (
                self.phases[current_phase]['epochs'][0],
                epoch
            )
            next_phase['epochs'] = (
                epoch + 1,
                next_phase['epochs'][1]
            )
            print(f" Advanced to {next_phase['name']}: {next_phase['description']}")


if __name__ == '__main__':
    # Test curriculum scheduler
    import yaml
    
    # Load sample config
    config_str = """
    enabled: true
    phase1:
      epochs: [1, 20]
      description: "Exposure stabilization"
      loss_weights:
        w_color: 0.001
        w_semantic: 0.0
        w_adv: 0.0
        w_freq: 0.0
        w_task: 0.3
        w_perceptual: 0.1
    phase2:
      epochs: [21, 50]
      description: "Semantic introduction"
      loss_weights:
        w_color: 0.001
        w_semantic: [0.0, 0.5]
        w_adv: 0.0
        w_freq: 0.00001
        w_task: 0.3
        w_perceptual: [0.1, 0.5]
    """
    
    config = yaml.safe_load(config_str)
    scheduler = CurriculumScheduler(config)
    
    # Test weight interpolation
    for epoch in [0, 10, 20, 25, 30, 40, 49]:
        weights = scheduler.get_phase_weights(epoch)
        phase_info = scheduler.check_phase_transition(epoch)
        print(f"Epoch {epoch+1}: Phase {phase_info['current_phase']} - w_semantic={weights['w_semantic']:.3f}")