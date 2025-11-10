"""
Real-time Training Monitor
Monitors training progress and alerts on anomalies
"""

import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class TrainingMonitor:
    def __init__(self, log_dir, refresh_interval=10):
        self.log_dir = Path(log_dir)
        self.refresh_interval = refresh_interval
        self.metrics_file = self.log_dir / 'metrics.csv'
        self.training_file = self.log_dir / 'training_log.csv'

        # Alert thresholds
        self.thresholds = {
            'psnr_drop': 5.0,        # Alert if PSNR drops >5 dB
            'loss_spike': 3.0,       # Alert if loss >3x recent average
            'delta_e_high': 25.0,    # Alert if Delta E >25
            'speed_drop': 0.3        # Alert if speed drops >30%
        }

    def load_metrics(self):
        """Load latest metrics"""
        try:
            if self.metrics_file.exists():
                df = pd.read_csv(self.metrics_file)
                return df
            return None
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None

    def load_training_log(self):
        """Load latest training log"""
        try:
            if self.training_file.exists():
                df = pd.read_csv(self.training_file)
                return df
            return None
        except Exception as e:
            print(f"Error loading training log: {e}")
            return None

    def check_anomalies(self, metrics_df, training_df):
        """Check for training anomalies"""
        alerts = []

        if metrics_df is not None and len(metrics_df) > 50:
            recent = metrics_df.tail(50)
            latest = metrics_df.tail(1)

            # Check PSNR drop
            psnr_mean = recent['psnr'].mean()
            psnr_current = latest['psnr'].values[0]
            if psnr_mean - psnr_current > self.thresholds['psnr_drop']:
                alerts.append(f"⚠️ PSNR dropped {psnr_mean - psnr_current:.2f} dB!")

            # Check Delta E
            delta_e_current = latest['delta_e'].values[0]
            if delta_e_current > self.thresholds['delta_e_high']:
                alerts.append(f"⚠️ High Delta E: {delta_e_current:.2f}")

            # Check speed
            speed_mean = recent['speed_imgs_sec'].mean()
            speed_current = latest['speed_imgs_sec'].values[0]
            if speed_current < speed_mean * (1 - self.thresholds['speed_drop']):
                alerts.append(f"⚠️ Training slowed: {speed_current:.1f} img/s (avg: {speed_mean:.1f})")

        if training_df is not None and len(training_df) > 50:
            recent_train = training_df.tail(50)
            latest_train = training_df.tail(1)

            # Check loss spike
            gen_loss_mean = recent_train['gen_loss'].mean()
            gen_loss_current = latest_train['gen_loss'].values[0]
            if gen_loss_current > gen_loss_mean * self.thresholds['loss_spike']:
                alerts.append(f"⚠️ Loss spike: {gen_loss_current:.4f} (avg: {gen_loss_mean:.4f})")

        return alerts

    def print_status(self):
        """Print current training status"""
        metrics_df = self.load_metrics()
        training_df = self.load_training_log()

        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 80)
        print("🔍 SemRoCL Training Monitor")
        print("=" * 80)
        print(f"Log directory: {self.log_dir}")
        print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Check anomalies
        alerts = self.check_anomalies(metrics_df, training_df)
        if alerts:
            print("🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")
            print()
        else:
            print("✅ No anomalies detected")
            print()

        # Metrics summary
        if metrics_df is not None and len(metrics_df) > 0:
            latest = metrics_df.tail(1)
            recent_50 = metrics_df.tail(50)

            print("📊 LATEST METRICS (Step {})".format(latest['step'].values[0]))
            print("-" * 80)
            print(f"  PSNR:      {latest['psnr'].values[0]:6.2f} dB    (avg last 50: {recent_50['psnr'].mean():6.2f})")
            print(f"  SSIM:      {latest['ssim'].values[0]:6.4f}      (avg last 50: {recent_50['ssim'].mean():6.4f})")
            print(f"  Delta E:   {latest['delta_e'].values[0]:6.2f}      (avg last 50: {recent_50['delta_e'].mean():6.2f})")
            print(f"  Speed:     {latest['speed_imgs_sec'].values[0]:6.2f} img/s (avg last 50: {recent_50['speed_imgs_sec'].mean():6.2f})")
            print(f"  Phase:     {latest['phase'].values[0]}")
            print()

        # Training summary
        if training_df is not None and len(training_df) > 0:
            latest_train = training_df.tail(1)
            recent_train_50 = training_df.tail(50)

            print("📈 LATEST LOSSES (Epoch {}, Batch {})".format(
                latest_train['epoch'].values[0],
                latest_train['batch'].values[0]
            ))
            print("-" * 80)
            print(f"  Generator:    {latest_train['gen_loss'].values[0]:8.4f}  (avg: {recent_train_50['gen_loss'].mean():8.4f})")
            print(f"  Discriminator:{latest_train['disc_loss'].values[0]:8.4f}  (avg: {recent_train_50['disc_loss'].mean():8.4f})")
            print(f"  Color:        {latest_train['color_loss'].values[0]:8.4f}")
            print(f"  Semantic:     {latest_train['semantic_loss'].values[0]:8.4f}")
            print(f"  Perceptual:   {latest_train['perceptual_loss'].values[0]:8.4f}")
            print(f"  Learning Rate:{latest_train['lr'].values[0]:8.6f}")
            print()

        # Progress estimation
        if metrics_df is not None and len(metrics_df) > 0:
            current_epoch = latest['epoch'].values[0]
            total_epochs = 200  # From config
            progress = (current_epoch / total_epochs) * 100

            print("⏱️  PROGRESS")
            print("-" * 80)
            print(f"  Current epoch: {current_epoch}/{total_epochs} ({progress:.1f}%)")

            # Estimate remaining time based on recent speed
            if len(metrics_df) > 100:
                recent_times = metrics_df.tail(100)
                avg_time_per_step = recent_times['iter_time_ms'].mean() / 1000  # Convert to seconds
                # Rough estimate: assume ~2500 steps per epoch
                remaining_epochs = total_epochs - current_epoch
                estimated_hours = (remaining_epochs * 2500 * avg_time_per_step) / 3600
                print(f"  Estimated remaining: {estimated_hours:.1f} hours")

            print()

        print("=" * 80)
        print(f"Refreshing every {self.refresh_interval}s... (Ctrl+C to exit)")

    def run(self):
        """Run monitoring loop"""
        print("Starting training monitor...")
        print(f"Watching: {self.log_dir}")
        print()

        try:
            while True:
                self.print_status()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")

    def plot_metrics(self, save_path=None):
        """Generate comprehensive metrics plots"""
        metrics_df = self.load_metrics()
        training_df = self.load_training_log()

        if metrics_df is None or len(metrics_df) == 0:
            print("No metrics data available")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('SemRoCL Training Metrics', fontsize=16, fontweight='bold')

        # PSNR over time
        axes[0, 0].plot(metrics_df['step'], metrics_df['psnr'], alpha=0.6, linewidth=0.5)
        axes[0, 0].plot(metrics_df['step'], metrics_df['psnr'].rolling(50).mean(),
                       color='red', linewidth=2, label='Moving Avg (50)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # SSIM over time
        axes[0, 1].plot(metrics_df['step'], metrics_df['ssim'], alpha=0.6, linewidth=0.5)
        axes[0, 1].plot(metrics_df['step'], metrics_df['ssim'].rolling(50).mean(),
                       color='red', linewidth=2, label='Moving Avg (50)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('Structural Similarity Index')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Delta E over time
        axes[1, 0].plot(metrics_df['step'], metrics_df['delta_e'], alpha=0.6, linewidth=0.5)
        axes[1, 0].plot(metrics_df['step'], metrics_df['delta_e'].rolling(50).mean(),
                       color='red', linewidth=2, label='Moving Avg (50)')
        axes[1, 0].axhline(y=15, color='green', linestyle='--', label='Target (<15)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Delta E')
        axes[1, 0].set_title('Color Difference (Delta E)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Training speed
        axes[1, 1].plot(metrics_df['step'], metrics_df['speed_imgs_sec'], alpha=0.6, linewidth=0.5)
        axes[1, 1].plot(metrics_df['step'], metrics_df['speed_imgs_sec'].rolling(50).mean(),
                       color='red', linewidth=2, label='Moving Avg (50)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Images/sec')
        axes[1, 1].set_title('Training Speed')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Loss weights over time
        if 'w_semantic' in metrics_df.columns:
            axes[2, 0].plot(metrics_df['step'], metrics_df['w_color'], label='Color', alpha=0.7)
            axes[2, 0].plot(metrics_df['step'], metrics_df['w_semantic'], label='Semantic', alpha=0.7)
            axes[2, 0].plot(metrics_df['step'], metrics_df['w_perceptual'], label='Perceptual', alpha=0.7)
            if 'w_adv' in metrics_df.columns:
                axes[2, 0].plot(metrics_df['step'], metrics_df['w_adv'], label='Adversarial', alpha=0.7)
            axes[2, 0].set_xlabel('Step')
            axes[2, 0].set_ylabel('Weight')
            axes[2, 0].set_title('Loss Weights (Curriculum)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # Losses from training log
        if training_df is not None and len(training_df) > 0:
            steps = range(len(training_df))
            axes[2, 1].plot(steps, training_df['gen_loss'], label='Generator', alpha=0.5, linewidth=0.5)
            axes[2, 1].plot(steps, pd.Series(training_df['gen_loss']).rolling(50).mean(),
                           color='blue', linewidth=2, label='Gen (Avg)')
            if 'disc_loss' in training_df.columns:
                disc_nonzero = training_df[training_df['disc_loss'] > 0]
                if len(disc_nonzero) > 0:
                    axes[2, 1].scatter(disc_nonzero.index, disc_nonzero['disc_loss'],
                                      s=1, alpha=0.3, color='orange', label='Discriminator')
            axes[2, 1].set_xlabel('Training Step')
            axes[2, 1].set_ylabel('Loss')
            axes[2, 1].set_title('Generator & Discriminator Loss')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Monitor SemRoCL training')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to training log directory')
    parser.add_argument('--refresh', type=int, default=10,
                       help='Refresh interval in seconds (default: 10)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots instead of monitoring')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save plot (if --plot is used)')

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir, args.refresh)

    if args.plot:
        save_path = args.save_plot or str(Path(args.log_dir) / 'training_metrics.png')
        monitor.plot_metrics(save_path)
    else:
        monitor.run()


if __name__ == '__main__':
    main()
