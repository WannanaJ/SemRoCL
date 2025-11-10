"""
Evaluation metrics driver for the SemRoCL TIP submission.

Metrics covered: PSNR, SSIM, LPIPS, Delta E, NIQE, BRISQUE, and HPI.
"""

import os
import sys
import csv
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_utils import (
    compute_psnr,
    compute_ssim,
    compute_delta_e,
    maybe_compute_niqe,
    maybe_compute_brisque
)

# LPIPS Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")


# ============================================
# Logging Configuration
# ============================================

def setup_logger(output_dir: str) -> logging.Logger:
    """ Setup logger"""
    log_file = os.path.join(output_dir, f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('eval_metrics')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # handler File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # handler Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================
# Image Loading
# ============================================

def load_image_as_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """
    tensor Load image as tensor

    Args:
        image_path:  Image path
        device:  Device

    Returns:
        Tensor with shape (1, C, H, W), normalized to [0, 1]
    """
    # PIL Load with PIL
    img = Image.open(image_path).convert('RGB')

    # numpy Convert to numpy
    img_array = np.array(img).astype(np.float32) / 255.0

    # tensor Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)

    return img_tensor.to(device)


# ============================================
# (HPI) Human Perception Index
# ============================================

def compute_hpi(
    psnr: float,
    ssim: float,
    lpips: float,
    niqe: float,
    delta_e: float
) -> float:
    """
     (HPI) Compute Human Perception Index (HPI)

    HPI
    HPI is a metric that combines multiple evaluation metrics

     Formula:
        HPI = 0.2 * (PSNR/40) + 0.3 * SSIM + 0.3 * (1 - LPIPS) + 0.1 * (1 - NIQE/10) + 0.1 * (1 - DeltaE/50)

     [0, 1]HPI
    All terms normalized to [0, 1], higher HPI indicates better visual quality

    Args:
        psnr: PSNR PSNR value
        ssim: SSIM SSIM value
        lpips: LPIPS LPIPS value
        niqe: NIQE NIQE value
        delta_e: Delta E Delta E value

    Returns:
        HPI [0, 1] HPI value, range [0, 1]
    """
    # PSNR [0, 1]40 Normalize PSNR to [0, 1], assume max 40
    psnr_norm = min(psnr / 40.0, 1.0)

    # SSIM [0, 1] SSIM already in [0, 1]
    ssim_norm = ssim

    # LPIPS [0, 1]1-LPIPS LPIPS in [0, 1], lower is better
    lpips_norm = 1.0 - lpips

    # NIQE NIQE lower is better, normalize
    niqe_norm = max(1.0 - niqe / 10.0, 0.0)

    # Delta E Delta E lower is better
    delta_e_norm = max(1.0 - delta_e / 50.0, 0.0)

    # Weighted sum
    hpi = (
        0.2 * psnr_norm +
        0.3 * ssim_norm +
        0.3 * lpips_norm +
        0.1 * niqe_norm +
        0.1 * delta_e_norm
    )

    return hpi


# ============================================
# Metrics Calculator Class
# ============================================

class MetricsCalculator:
    """ Evaluation metrics calculator"""

    def __init__(self, device: torch.device, logger: logging.Logger):
        """
         Initialize

        Args:
            device:  Computing device
            logger:  Logger
        """
        self.device = device
        self.logger = logger

        # LPIPS Initialize LPIPS model
        self.lpips_model = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                self.lpips_model.eval()
                self.logger.info("LPIPS model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load LPIPS model: {e}")

    def compute_all_metrics(
        self,
        enhanced_path: str,
        target_path: str
    ) -> Dict[str, float]:
        """
         Compute all metrics

        Args:
            enhanced_path:  Enhanced image path
            target_path:  Target image path

        Returns:
             Metrics dict
        """
        metrics = {}

        # Load images
        try:
            enhanced = load_image_as_tensor(enhanced_path, self.device)
            target = load_image_as_tensor(target_path, self.device)

            # Ensure size match
            if enhanced.shape != target.shape:
                enhanced = F.interpolate(
                    enhanced,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
        except Exception as e:
            self.logger.error(f"Failed to load images: {e}")
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'lpips': 1.0,
                'niqe': 10.0,
                'brisque': 100.0,
                'delta_e': 50.0,
                'hpi': 0.0,
                'error': str(e)
            }

        # 1. PSNR
        try:
            metrics['psnr'] = compute_psnr(enhanced, target)
        except Exception as e:
            self.logger.warning(f"Failed to compute PSNR: {e}")
            metrics['psnr'] = 0.0

        # 2. SSIM
        try:
            metrics['ssim'] = compute_ssim(enhanced, target)
        except Exception as e:
            self.logger.warning(f"Failed to compute SSIM: {e}")
            metrics['ssim'] = 0.0

        # 3. LPIPS
        if self.lpips_model is not None:
            try:
                with torch.no_grad():
                    # LPIPS [-1, 1]  LPIPS needs images in [-1, 1] range
                    enhanced_norm = enhanced * 2.0 - 1.0
                    target_norm = target * 2.0 - 1.0
                    lpips_value = self.lpips_model(enhanced_norm, target_norm)
                    metrics['lpips'] = float(lpips_value.mean().item())
            except Exception as e:
                self.logger.warning(f"Failed to compute LPIPS: {e}")
                metrics['lpips'] = 1.0
        else:
            metrics['lpips'] = 1.0  # Default value

        # 4. NIQE () NIQE (no-reference, only enhanced)
        try:
            metrics['niqe'] = maybe_compute_niqe(enhanced)
        except Exception as e:
            self.logger.warning(f"Failed to compute NIQE: {e}")
            metrics['niqe'] = 10.0

        # 5. BRISQUE () BRISQUE (no-reference, only enhanced)
        try:
            metrics['brisque'] = maybe_compute_brisque(enhanced)
        except Exception as e:
            self.logger.warning(f"Failed to compute BRISQUE: {e}")
            metrics['brisque'] = 100.0

        # 6. Delta E () Delta E (color difference)
        try:
            metrics['delta_e'] = compute_delta_e(enhanced, target)
        except Exception as e:
            self.logger.warning(f"Failed to compute Delta E: {e}")
            metrics['delta_e'] = 50.0

        # 7. HPI () HPI (Human Perception Index)
        try:
            metrics['hpi'] = compute_hpi(
                psnr=metrics['psnr'],
                ssim=metrics['ssim'],
                lpips=metrics['lpips'],
                niqe=metrics['niqe'],
                delta_e=metrics['delta_e']
            )
        except Exception as e:
            self.logger.warning(f"Failed to compute HPI: {e}")
            metrics['hpi'] = 0.0

        return metrics


# ============================================
# Ablation Metrics Computation
# ============================================

def compute_ablation_metrics(
    ablation_dir: str,
    target_dir: str,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
     Compute metrics for single ablation

    Args:
        ablation_dir:  Ablation images directory
        target_dir:  Target images directory
        device:  Computing device
        logger:  Logger

    Returns:
        (, ) (Average metrics, Per-image metrics list)
    """
    calculator = MetricsCalculator(device, logger)

    # Get all enhanced images
    enhanced_images = sorted([
        f for f in os.listdir(ablation_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not enhanced_images:
        logger.warning(f"No images found in {ablation_dir}")
        return {}, []

    logger.info(f"Computing metrics for {len(enhanced_images)} images...")

    per_image_metrics = []
    accumulated_metrics = {
        'psnr': [], 'ssim': [], 'lpips': [],
        'niqe': [], 'brisque': [], 'delta_e': [], 'hpi': []
    }

    # Iterate each image
    for img_name in tqdm(enhanced_images, desc="Computing metrics"):
        enhanced_path = os.path.join(ablation_dir, img_name)
        target_path = os.path.join(target_dir, img_name)

        # Check if target exists
        if not os.path.exists(target_path):
            logger.warning(f"Target not found for {img_name}, skipping")
            continue

        # Compute metrics
        metrics = calculator.compute_all_metrics(enhanced_path, target_path)

        # Save per-image metrics
        metrics['filename'] = img_name
        per_image_metrics.append(metrics)

        # Accumulate for averaging
        for key in accumulated_metrics.keys():
            if key in metrics and not isinstance(metrics[key], str):
                accumulated_metrics[key].append(metrics[key])

    # Compute averages
    avg_metrics = {}
    for key, values in accumulated_metrics.items():
        if values:
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f'{key}_std'] = float(np.std(values))
        else:
            avg_metrics[key] = 0.0
            avg_metrics[f'{key}_std'] = 0.0

    logger.info(f"Average metrics: PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}, "
                f"LPIPS={avg_metrics['lpips']:.4f}, HPI={avg_metrics['hpi']:.4f}")

    return avg_metrics, per_image_metrics


# ============================================
# Main Evaluation Pipeline
# ============================================

def evaluate_all_ablations(config_path: str) -> None:
    """
     Evaluate all ablation experiments

    Args:
        config_path:  Configuration file path
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    output_base = config['output']['base_dir']

    # Setup logger
    logger = setup_logger(output_base)
    logger.info("=" * 80)
    logger.info("SemRoCL Metrics Evaluation for TIP Journal Submission")
    logger.info("=" * 80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Get target images directory
    dataset_name = config['experiment']['dataset']
    if dataset_name == 'LOL-v1':
        target_dir = os.path.join(config['data']['data_dir'], 'LOL-v1', 'eval15', 'high')
    elif dataset_name == 'LOL-v2-real':
        target_dir = os.path.join(config['data']['data_dir'], 'LOL-v2', 'Real_captured', 'Test', 'Normal')
    elif dataset_name == 'LOL-v2-synthetic':
        target_dir = os.path.join(config['data']['data_dir'], 'LOL-v2', 'Synthetic', 'Test', 'Normal')
    else:
        target_dir = os.path.join(config['data']['data_dir'], dataset_name, 'test', 'high')

    logger.info(f"Target images directory: {target_dir}")

    if not os.path.exists(target_dir):
        logger.error(f"Target directory not found: {target_dir}")
        return

    # Get all ablation directories
    ablation_dirs = [
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and
        os.path.exists(os.path.join(output_base, d, 'images'))
    ]

    if not ablation_dirs:
        logger.error("No ablation directories found!")
        return

    logger.info(f"Found {len(ablation_dirs)} ablation experiments to evaluate")

    # Evaluate each ablation
    all_results = []

    for ablation_name in sorted(ablation_dirs):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating: {ablation_name}")
        logger.info(f"{'=' * 80}")

        ablation_images_dir = os.path.join(output_base, ablation_name, 'images')

        try:
            avg_metrics, per_image_metrics = compute_ablation_metrics(
                ablation_dir=ablation_images_dir,
                target_dir=target_dir,
                device=device,
                logger=logger
            )

            # Save results
            result = {
                'ablation_name': ablation_name,
                **avg_metrics
            }
            all_results.append(result)

            # Save per-image detailed metrics
            if config['output'].get('save_detailed', True):
                detailed_csv = os.path.join(output_base, ablation_name, 'detailed_metrics.csv')
                with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
                    if per_image_metrics:
                        writer = csv.DictWriter(f, fieldnames=per_image_metrics[0].keys())
                        writer.writeheader()
                        writer.writerows(per_image_metrics)
                logger.info(f"Detailed metrics saved to: {detailed_csv}")

        except Exception as e:
            logger.error(f"Failed to evaluate {ablation_name}: {e}")
            all_results.append({
                'ablation_name': ablation_name,
                'error': str(e)
            })

    # Save summary results
    if config['output'].get('save_csv', True):
        summary_csv = os.path.join(output_base, 'metrics_summary.csv')
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            if all_results:
                fieldnames = ['ablation_name', 'psnr', 'psnr_std', 'ssim', 'ssim_std',
                             'lpips', 'lpips_std', 'niqe', 'niqe_std', 'brisque', 'brisque_std',
                             'delta_e', 'delta_e_std', 'hpi', 'hpi_std']
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_results)
        logger.info(f"\nSummary metrics saved to: {summary_csv}")

    # Generate readable report
    report_path = os.path.join(output_base, 'visual_quality_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SemRoCL Ablation Study - Visual Quality Report\n")
        f.write("TIP Journal Submission\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of ablations: {len(all_results)}\n\n")

        f.write("=" * 100 + "\n")
        f.write("Metrics Summary (sorted by HPI)\n")
        f.write("=" * 100 + "\n\n")

        # HPI Sort by HPI
        sorted_results = sorted(
            [r for r in all_results if 'hpi' in r],
            key=lambda x: x.get('hpi', 0),
            reverse=True
        )

        f.write(f"{'Rank':<6} {'Ablation Name':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10} {'HPI':<10}\n")
        f.write("-" * 100 + "\n")

        for rank, result in enumerate(sorted_results, 1):
            f.write(
                f"{rank:<6} {result['ablation_name']:<30} "
                f"{result.get('psnr', 0):<10.2f} {result.get('ssim', 0):<10.4f} "
                f"{result.get('lpips', 0):<10.4f} {result.get('hpi', 0):<10.4f}\n"
            )

        f.write("\n" + "=" * 100 + "\n")
        f.write("Detailed Metrics\n")
        f.write("=" * 100 + "\n\n")

        for rank, result in enumerate(sorted_results, 1):
            f.write(f"\n{rank}. {result['ablation_name']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  PSNR:     {result.get('psnr', 0):.2f}  {result.get('psnr_std', 0):.2f} dB\n")
            f.write(f"  SSIM:     {result.get('ssim', 0):.4f}  {result.get('ssim_std', 0):.4f}\n")
            f.write(f"  LPIPS:    {result.get('lpips', 0):.4f}  {result.get('lpips_std', 0):.4f}\n")
            f.write(f"  NIQE:     {result.get('niqe', 0):.4f}  {result.get('niqe_std', 0):.4f}\n")
            f.write(f"  BRISQUE:  {result.get('brisque', 0):.4f}  {result.get('brisque_std', 0):.4f}\n")
            f.write(f"  Delta E:  {result.get('delta_e', 0):.4f}  {result.get('delta_e_std', 0):.4f}\n")
            f.write(f"  HPI:      {result.get('hpi', 0):.4f}  {result.get('hpi_std', 0):.4f}\n")

    logger.info(f"Visual quality report saved to: {report_path}")
    logger.info("\n" + "=" * 80)
    logger.info("Metrics evaluation completed!")
    logger.info("=" * 80)


# ============================================
# Command Line Interface
# ============================================

def parse_args():
    """ Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate metrics for SemRoCL ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file (YAML)'
    )

    return parser.parse_args()


def main():
    """ Main function"""
    args = parse_args()
    evaluate_all_ablations(args.config)


if __name__ == '__main__':
    main()
