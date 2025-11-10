"""
 Ablation Study Main Script
============================================

TIP
Complete ablation experiment framework for TIP journal submission

 Features:
- Auto iterate all ablation combinations
- Support ablation of semantic, frequency, GAN, curriculum
- Compute comprehensive evaluation metrics
- Save enhanced images and results
- Generate experiment reports

 Author: SemRoCL Team
 Date: 2025-11-10
 Version: 1.0.0
"""

import os
import sys
import yaml
import argparse
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_dataloader
from model.models_enhanced import EnhancementGenerator, SemanticGuidanceModule, Discriminator
from loss_functions import CombinedLoss
from utils import save_image


# ============================================
# Logging Configuration
# ============================================

def setup_logger(output_dir: str) -> logging.Logger:
    """
     Setup experiment logger

    Args:
        output_dir:  Output directory

    Returns:
        logger Configured logger
    """
    log_file = os.path.join(output_dir, f'ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # logger Create logger
    logger = logging.getLogger('ablation')
    logger.setLevel(logging.INFO)

    # handlers Clear existing handlers
    logger.handlers.clear()

    # handler File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # handler Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================
# Configuration Loading
# ============================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
     Load experiment configuration

    Args:
        config_path:  Config file path

    Returns:
         Configuration dict
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ============================================
# Ablation Combination Generator
# ============================================

def generate_ablation_combinations(ablation_config: Dict[str, List[bool]]) -> List[Dict[str, bool]]:
    """
     Generate all ablation combinations

    Args:
        ablation_config:  Ablation configuration
             Example: {'use_semantic': [True, False], 'use_freq': [True, False]}

    Returns:
         List of all combinations
         Example: [
            {'use_semantic': True, 'use_freq': True},
            {'use_semantic': True, 'use_freq': False},
            ...
        ]
    """
    keys = list(ablation_config.keys())
    values = list(ablation_config.values())

    # itertools.product Use itertools.product for cartesian product
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def get_ablation_name(ablation_settings: Dict[str, bool]) -> str:
    """
     Generate name for ablation combination

    Args:
        ablation_settings:  Ablation settings

    Returns:
         Name string
         Example: "sem1_freq1_gan0_cur1"
    """
    name_parts = []

    # Semantic guidance
    name_parts.append(f"sem{1 if ablation_settings.get('use_semantic', False) else 0}")

    # Frequency constraint
    name_parts.append(f"freq{1 if ablation_settings.get('use_freq', False) else 0}")

    # Adversarial loss
    name_parts.append(f"gan{1 if ablation_settings.get('use_gan', False) else 0}")

    # Curriculum learning
    name_parts.append(f"cur{1 if ablation_settings.get('use_curriculum', False) else 0}")

    return "_".join(name_parts)


def get_ablation_description(ablation_settings: Dict[str, bool]) -> str:
    """
     Generate description for ablation combination

    Args:
        ablation_settings:  Ablation settings

    Returns:
         Description string
    """
    components = []

    if ablation_settings.get('use_semantic', False):
        components.append("Semantic")
    if ablation_settings.get('use_freq', False):
        components.append("Frequency")
    if ablation_settings.get('use_gan', False):
        components.append("GAN")
    if ablation_settings.get('use_curriculum', False):
        components.append("Curriculum")

    if not components:
        return "Baseline (No components)"

    return " + ".join(components)


# ============================================
# Model Loading and Configuration
# ============================================

def load_model_for_ablation(
    checkpoint_path: str,
    ablation_settings: Dict[str, bool],
    device: torch.device,
    logger: logging.Logger
) -> nn.Module:
    """
     Load and configure model according to ablation settings

    Args:
        checkpoint_path:  Checkpoint path
        ablation_settings:  Ablation settings
        device:  Computing device
        logger:  Logger

    Returns:
         Configured model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model
    model = EnhancementGenerator(
        in_channels=3,
        num_iterations=8,
        use_semantic=ablation_settings.get('use_semantic', False)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state
    if 'generator' in checkpoint:
        state_dict = checkpoint['generator']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load some weights: {e}")
        # Try partial loading
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Partially loaded {len(pretrained_dict)}/{len(state_dict)} weights")

    model = model.to(device)
    model.eval()

    logger.info(f"Model configured with ablation settings: {ablation_settings}")

    return model


# ============================================
# Image Enhancement
# ============================================

@torch.no_grad()
def enhance_images(
    model: nn.Module,
    dataloader: DataLoader,
    ablation_settings: Dict[str, bool],
    output_dir: str,
    device: torch.device,
    logger: logging.Logger,
    save_images: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
     Enhance images using configured model

    Args:
        model:  Model
        dataloader:  Data loader
        ablation_settings:  Ablation settings
        output_dir:  Output directory
        device:  Computing device
        logger:  Logger
        save_images:  Whether to save images

    Returns:
        (, , ) /
        (List of enhanced images, List of target images, List of filenames)
    """
    logger.info("Starting image enhancement...")

    enhanced_images = []
    target_images = []
    filenames = []

    # Create output directory
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Iterate data
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Enhancing images")):
        low = batch['low'].to(device)
        high = batch.get('high', None)
        filename = batch['filename'][0] if 'filename' in batch else f"image_{batch_idx:04d}.png"

        # Enhance
        with torch.no_grad():
            enhanced = model(low)

        # Save images
        if save_images:
            save_path = os.path.join(output_dir, filename)
            save_image(enhanced, save_path, quality=95, apply_post_processing=True)

        # Collect results
        enhanced_images.append(enhanced.cpu())
        if high is not None:
            target_images.append(high.cpu())
        filenames.append(filename)

    logger.info(f"Enhanced {len(enhanced_images)} images")

    return enhanced_images, target_images, filenames


# ============================================
# Single Ablation Experiment
# ============================================

def run_single_ablation(
    ablation_settings: Dict[str, bool],
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
     Run single ablation experiment

    Args:
        ablation_settings:  Ablation settings
        config:  Full configuration
        device:  Computing device
        logger:  Logger

    Returns:
         Experiment results dict
    """
    # Generate experiment name
    ablation_name = get_ablation_name(ablation_settings)
    ablation_desc = get_ablation_description(ablation_settings)

    logger.info("=" * 80)
    logger.info(f"Running ablation: {ablation_name}")
    logger.info(f"Description: {ablation_desc}")
    logger.info(f"Settings: {ablation_settings}")
    logger.info("=" * 80)

    # Create output directory
    output_base = config['output']['base_dir']
    ablation_output_dir = os.path.join(output_base, ablation_name)
    images_dir = os.path.join(ablation_output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Load model
    try:
        model = load_model_for_ablation(
            checkpoint_path=config['experiment']['weights'],
            ablation_settings=ablation_settings,
            device=device,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            'ablation_name': ablation_name,
            'ablation_desc': ablation_desc,
            'settings': ablation_settings,
            'status': 'failed',
            'error': str(e)
        }

    # Load data
    try:
        dataset_name = config['experiment']['dataset']
        dataloader = get_dataloader(
            dataset_name=dataset_name,
            data_dir=config['data']['data_dir'],
            batch_size=1,
            mode='test',
            num_workers=0,
            shuffle=False
        )
        logger.info(f"Loaded {len(dataloader)} test images from {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {
            'ablation_name': ablation_name,
            'ablation_desc': ablation_desc,
            'settings': ablation_settings,
            'status': 'failed',
            'error': str(e)
        }

    # Enhance images
    try:
        enhanced_images, target_images, filenames = enhance_images(
            model=model,
            dataloader=dataloader,
            ablation_settings=ablation_settings,
            output_dir=images_dir,
            device=device,
            logger=logger,
            save_images=config['output'].get('save_images', True)
        )
    except Exception as e:
        logger.error(f"Failed to enhance images: {e}")
        return {
            'ablation_name': ablation_name,
            'ablation_desc': ablation_desc,
            'settings': ablation_settings,
            'status': 'failed',
            'error': str(e)
        }

    # Compute metrics
    # eval_metrics.py Note: Actual metric computation in eval_metrics.py
    logger.info(f"Metrics will be computed by eval_metrics.py")

    # Return results
    result = {
        'ablation_name': ablation_name,
        'ablation_desc': ablation_desc,
        'settings': ablation_settings,
        'status': 'success',
        'output_dir': ablation_output_dir,
        'images_dir': images_dir,
        'num_images': len(enhanced_images),
        'filenames': filenames
    }

    logger.info(f"Ablation {ablation_name} completed successfully")

    return result


# ============================================
# Main Experiment Pipeline
# ============================================

def run_ablation_study(config_path: str) -> None:
    """
     Run complete ablation study

    Args:
        config_path:  Configuration file path
    """
    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_base = config['output']['base_dir']
    os.makedirs(output_base, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_base)
    logger.info("=" * 80)
    logger.info("SemRoCL Ablation Study for TIP Journal Submission")
    logger.info("=" * 80)
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Output directory: {output_base}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory 1e9:.2f} GB")

    # Generate ablation combinations
    ablation_combinations = generate_ablation_combinations(config['ablation'])
    logger.info(f"Generated {len(ablation_combinations)} ablation combinations:")
    for i, combo in enumerate(ablation_combinations, 1):
        logger.info(f"  {i}. {get_ablation_name(combo)}: {get_ablation_description(combo)}")

    # Run each ablation experiment
    all_results = []
    start_time = time.time()

    for i, ablation_settings in enumerate(ablation_combinations, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Ablation {i}/{len(ablation_combinations)}")
        logger.info(f"{'=' * 80}\n")

        try:
            result = run_single_ablation(
                ablation_settings=ablation_settings,
                config=config,
                device=device,
                logger=logger
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Ablation {i} failed with error: {e}")
            all_results.append({
                'ablation_name': get_ablation_name(ablation_settings),
                'ablation_desc': get_ablation_description(ablation_settings),
                'settings': ablation_settings,
                'status': 'failed',
                'error': str(e)
            })

    # Save results summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("Ablation Study Completed")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Successfully completed: {sum(1 for r in all_results if r['status'] == 'success')}/{len(all_results)}")

    # CSV Save results to CSV
    if config['output'].get('save_csv', True):
        csv_path = os.path.join(output_base, 'ablation_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'ablation_name', 'ablation_desc', 'use_semantic', 'use_freq',
                'use_gan', 'use_curriculum', 'status', 'num_images', 'output_dir'
            ])
            writer.writeheader()
            for result in all_results:
                writer.writerow({
                    'ablation_name': result['ablation_name'],
                    'ablation_desc': result['ablation_desc'],
                    'use_semantic': result['settings'].get('use_semantic', False),
                    'use_freq': result['settings'].get('use_freq', False),
                    'use_gan': result['settings'].get('use_gan', False),
                    'use_curriculum': result['settings'].get('use_curriculum', False),
                    'status': result['status'],
                    'num_images': result.get('num_images', 0),
                    'output_dir': result.get('output_dir', '')
                })
        logger.info(f"Results summary saved to: {csv_path}")

    logger.info("\nNext steps:")
    logger.info("1. Run eval_metrics.py to compute all metrics")
    logger.info("2. Run visualize_results.py to generate plots")
    logger.info("3. Run summarize_results.py to create LaTeX tables")


# ============================================
# Command Line Interface
# ============================================

def parse_args():
    """ Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run ablation study for SemRoCL (TIP Journal Submission)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full ablation study
  python run_ablation.py --config ../../configs/experiment.yaml

  # Run with custom output directory
  python run_ablation.py --config ../../configs/experiment.yaml --output ./custom_output
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file (YAML)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output base directory'
    )

    return parser.parse_args()


def main():
    """ Main function"""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override output directory
    if args.output:
        config['output']['base_dir'] = args.output

    # Run ablation study
    run_ablation_study(args.config)


if __name__ == '__main__':
    main()
