"""
Automatic Dataset Structure Scanner (Enhanced Version)

Scans low-light image dataset directories and automatically detects:
- Low-light image directories
- High-quality reference directories
- Image count and formats
- Whether has paired references

Author: SemRoCL Team
Date: 2025-11-10
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Low-light directory patterns
LOW_DIR_PATTERNS = [
    'low', 'input', 'dark', 'short', 'lowlight', 'under', 'underexposed'
]

# High-quality / reference directory patterns
HIGH_DIR_PATTERNS = [
    'high', 'gt', 'groundtruth', 'ground_truth',
    'normal', 'target', 'reference', 'ref', 'long', 'well', 'welllit', 'well-lit'
]

# Directories to exclude completely
EXCLUDED_DIRS = ['__MACOSX', '__pycache__', '.git', '.vscode', 'node_modules']

# Prioritize these directory names for evaluation
EVAL_PRIORITY_PATTERNS = ['eval', 'test', 'val', 'validation']

# Deprioritize these directory names (training data)
TRAIN_PATTERNS = ['train', 'training']

# Image file extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


def setup_logger(debug: bool = False) -> logging.Logger:
    """Setup logger"""
    logger = logging.getLogger('dataset_scanner')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def is_image_file(filename: str) -> bool:
    """Check if file is an image"""
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)


def should_exclude_dir(dirname: str) -> bool:
    """Check if directory should be excluded"""
    return any(excl in dirname for excl in EXCLUDED_DIRS)


def count_images_in_dir(directory: str, recursive: bool = False) -> int:
    """Count images in a directory"""
    if not os.path.exists(directory):
        return 0

    count = 0
    if recursive:
        for root, _, files in os.walk(directory):
            # Skip excluded directories
            if any(should_exclude_dir(root) for _ in [1]):
                continue
            for file in files:
                if is_image_file(file):
                    count += 1
    else:
        try:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and is_image_file(file):
                    count += 1
        except Exception:
            pass

    return count


def matches_pattern(dirname: str, patterns: List[str]) -> bool:
    """Check if directory name matches any pattern (case-insensitive)"""
    dirname_lower = dirname.lower()
    return any(pattern.lower() in dirname_lower for pattern in patterns)


def has_eval_priority(dirname: str) -> bool:
    """Check if directory has evaluation priority"""
    dirname_lower = dirname.lower()
    return any(pattern in dirname_lower for pattern in EVAL_PRIORITY_PATTERNS)


def is_training_dir(dirname: str) -> bool:
    """Check if directory is training data"""
    dirname_lower = dirname.lower()
    return any(pattern in dirname_lower for pattern in TRAIN_PATTERNS)


def find_low_high_dirs_recursive(base_dir: str, logger: logging.Logger, max_depth: int = 3) -> Dict:
    """
    Recursively search for low/high directory pairs up to max_depth

    Returns:
        Dict with: {
            'low_dir': path,
            'high_dir': path or None,
            'low_count': int,
            'high_count': int,
            'depth': int
        }
    """
    # Store base_dir for relative path calculations
    base_dir_abs = os.path.abspath(base_dir)

    def search_recursive(current_path: str, current_depth: int) -> List[Dict]:
        """Recursively search for low/high directories"""
        if current_depth > max_depth:
            return []

        results = []

        try:
            entries = os.listdir(current_path)
        except Exception:
            return []

        # Check if current directory itself contains images (flat structure)
        direct_images = count_images_in_dir(current_path, recursive=False)

        # Find subdirectories
        subdirs = []
        for entry in entries:
            entry_path = os.path.join(current_path, entry)
            if os.path.isdir(entry_path) and not should_exclude_dir(entry):
                subdirs.append((entry, entry_path))

        # Check if current level has low/high pair
        low_dirs = [(name, path) for name, path in subdirs if matches_pattern(name, LOW_DIR_PATTERNS)]
        high_dirs = [(name, path) for name, path in subdirs if matches_pattern(name, HIGH_DIR_PATTERNS)]

        if low_dirs:
            # Found low directory at this level
            for low_name, low_path in low_dirs:
                low_count = count_images_in_dir(low_path, recursive=True)
                if low_count > 0:
                    # Look for corresponding high directory
                    high_path = None
                    high_count = 0

                    for high_name, high_p in high_dirs:
                        high_c = count_images_in_dir(high_p, recursive=True)
                        if high_c > 0:
                            high_path = high_p
                            high_count = high_c
                            break

                    # Check if any part of the RELATIVE path has eval priority or is training
                    low_path_rel = os.path.relpath(low_path, base_dir_abs)
                    has_priority = has_eval_priority(low_path_rel)
                    is_train = is_training_dir(low_path_rel)

                    results.append({
                        'low_dir': low_path,
                        'high_dir': high_path,
                        'low_count': low_count,
                        'high_count': high_count,
                        'depth': current_depth,
                        'has_eval_priority': has_priority,
                        'is_training': is_train
                    })

        elif direct_images > 0 and current_depth == 0:
            # Flat structure with images in current directory (root only)
            # Only accept if there are enough images (not just a stray screenshot)
            if direct_images >= 5:
                current_path_rel = os.path.relpath(current_path, base_dir_abs)
                results.append({
                    'low_dir': current_path,
                    'high_dir': None,
                    'low_count': direct_images,
                    'high_count': 0,
                    'depth': current_depth,
                    'has_eval_priority': has_eval_priority(current_path_rel),
                    'is_training': is_training_dir(current_path_rel)
                })

        # Recursively search subdirectories
        for subdir_name, subdir_path in subdirs:
            # Skip if this is a recognized low/high directory (already handled)
            if matches_pattern(subdir_name, LOW_DIR_PATTERNS + HIGH_DIR_PATTERNS):
                continue

            sub_results = search_recursive(subdir_path, current_depth + 1)
            results.extend(sub_results)

        return results

    # Start recursive search
    all_results = search_recursive(base_dir, 0)

    if not all_results:
        return None

    # Prioritize results: prefer eval directories, avoid training, prefer shallower depth, then more images
    def priority_score(result):
        return (
            1 if result['has_eval_priority'] else 0,  # Prefer eval/test directories
            0 if result['is_training'] else 1,  # Avoid training directories
            -result['depth'],  # Prefer shallower (negate for sorting)
            result['low_count']  # Prefer more images
        )

    all_results.sort(key=priority_score, reverse=True)

    # Debug: log all candidates and their scores (if logger available and multiple results)
    if len(all_results) > 1 and logger:
        logger.debug(f"Found {len(all_results)} candidates:")
        for i, res in enumerate(all_results[:3]):  # Show top 3
            score = priority_score(res)
            logger.debug(f"  {i+1}. {res['low_dir']} - score: {score}, count: {res['low_count']}")

    return all_results[0]


def scan_dataset_structure(dataset_path: str, logger: logging.Logger) -> Dict:
    """
    Scan a single dataset directory and identify its structure

    Returns:
        Dict with dataset metadata including paths, counts, and structure type
    """
    dataset_name = os.path.basename(dataset_path)

    logger.info(f"Scanning dataset: {dataset_name}")

    # Try recursive search for low/high directories
    result = find_low_high_dirs_recursive(dataset_path, logger)

    if result is None:
        logger.warning(f"  No images found in {dataset_name}")
        return {
            'name': dataset_name,
            'path': dataset_path,
            'low_dir': None,
            'high_dir': None,
            'has_reference': False,
            'low_count': 0,
            'high_count': 0,
            'structure_type': 'unknown'
        }

    # Determine structure type
    if result['high_dir'] is not None:
        structure_type = 'paired'
    elif result['depth'] == 0:
        structure_type = 'flat'
    else:
        structure_type = 'unpaired'

    # Convert absolute paths to relative paths from dataset root
    low_dir_rel = os.path.relpath(result['low_dir'], dataset_path) if result['low_dir'] else None
    high_dir_rel = os.path.relpath(result['high_dir'], dataset_path) if result['high_dir'] else None

    # Fix "." to empty string for cleaner config
    if low_dir_rel == '.':
        low_dir_rel = ''
    if high_dir_rel == '.':
        high_dir_rel = ''

    dataset_info = {
        'name': dataset_name,
        'path': dataset_path,
        'low_dir': low_dir_rel if low_dir_rel else '.',
        'high_dir': high_dir_rel if high_dir_rel else '',
        'has_reference': result['high_dir'] is not None,
        'low_count': result['low_count'],
        'high_count': result['high_count'],
        'structure_type': structure_type
    }

    logger.info(f"  Structure: {structure_type}")
    logger.info(f"  Low dir: {dataset_info['low_dir']} ({result['low_count']} images)")
    if result['high_dir']:
        logger.info(f"  High dir: {dataset_info['high_dir']} ({result['high_count']} images)")
    else:
        logger.info(f"  No reference images found")

    return dataset_info


def scan_all_datasets(base_dir: str, logger: logging.Logger) -> List[Dict]:
    """
    Scan all dataset subdirectories in base_dir

    Returns:
        List of dataset info dictionaries
    """
    datasets = []

    logger.info(f"Scanning base directory: {base_dir}")
    logger.info("=" * 100)

    if not os.path.exists(base_dir):
        logger.error(f"Base directory does not exist: {base_dir}")
        return datasets

    # Get all subdirectories
    try:
        entries = sorted(os.listdir(base_dir))
    except Exception as e:
        logger.error(f"Failed to list directory: {e}")
        return datasets

    for entry in entries:
        entry_path = os.path.join(base_dir, entry)

        # Only process directories, skip excluded ones
        if os.path.isdir(entry_path) and not should_exclude_dir(entry):
            dataset_info = scan_dataset_structure(entry_path, logger)
            if dataset_info['low_count'] > 0:  # Only include datasets with images
                datasets.append(dataset_info)
            logger.info("")

    logger.info("=" * 100)
    logger.info(f"Total datasets found: {len(datasets)}")

    return datasets


def generate_config_file(datasets: List[Dict], output_path: str, logger: logging.Logger) -> None:
    """
    Generate experiment configuration file from scanned datasets
    """
    logger.info(f"Generating config file: {output_path}")

    # Create config structure
    config = {
        'experiment': {
            'name': 'semrocl_ablation_multi_dataset',
            'description': 'Auto-generated multi-dataset ablation experiment config',
            'model': {
                'checkpoint': './outputs/semantic_enhancement_stage2_ENHANCED/checkpoints/best_model.pth',
                'architecture': 'EnhancedLowLightModel',
            },
            'datasets': []
        },

        'ablation': {
            'use_semantic': [True, False],
            'use_freq': [True, False],
            'use_gan': [True, False],
            'use_curriculum': [True, False]
        },

        'metrics': {
            'full_reference': ['psnr', 'ssim', 'lpips', 'delta_e'],
            'no_reference': ['niqe', 'brisque'],
            'combined': ['hpi']
        },

        'output': {
            'base_dir': './outputs/ablation_multi_dataset',
            'save_images': True,
            'save_csv': True,
            'save_tex': True,
            'save_plots': True,
            'image_format': 'png',
            'plot_dpi': 300
        }
    }

    # Add dataset entries
    for ds in datasets:
        dataset_entry = {
            'name': ds['name'],
            'path': ds['path'],
            'low_dir': ds['low_dir'],
            'high_dir': ds['high_dir'],
            'has_reference': ds['has_reference'],
            'enabled': True
        }
        config['experiment']['datasets'].append(dataset_entry)

    # Write config file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Config file saved: {output_path}")


def generate_report(datasets: List[Dict], output_path: str, logger: logging.Logger) -> None:
    """
    Generate human-readable scan report
    """
    logger.info(f"Generating report: {output_path}")

    total_images = sum(ds['low_count'] for ds in datasets)
    paired_count = sum(1 for ds in datasets if ds['has_reference'])
    unpaired_count = len(datasets) - paired_count

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("\n")
        f.write("Dataset Scan Summary Report\n")
        f.write("=" * 100 + "\n")
        f.write("\n")
        f.write(f" Scan Time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" Total Datasets: {len(datasets)}\n")
        f.write(f" With Reference: {paired_count}\n")
        f.write(f" Without Reference: {unpaired_count}\n")
        f.write(f" Total Images: {total_images}\n")
        f.write("\n")

        f.write("=" * 100 + "\n")
        f.write(" Details\n")
        f.write("=" * 100 + "\n")
        f.write("\n")

        for idx, ds in enumerate(datasets, 1):
            f.write(f"{idx}. {ds['name']}\n")
            f.write("-" * 100 + "\n")
            f.write(f"    Path: {ds['path']}\n")
            f.write(f"    Structure Type: {ds['structure_type']}\n")
            f.write(f"    Low Dir: {ds['low_dir']} ({ds['low_count']} images)\n")

            if ds['has_reference']:
                f.write(f"    High Dir: {ds['high_dir']} ({ds['high_count']} images)\n")
                f.write(f"    Has Reference: YES\n")
            else:
                f.write(f"    Has Reference: NO\n")

            f.write("\n")

        f.write("=" * 100 + "\n")
        f.write(" Recommendations\n")
        f.write("=" * 100 + "\n")
        f.write("\n")
        f.write(f"   {paired_count} datasets have reference images (can use full metrics)\n")
        f.write(f"   {unpaired_count} datasets have no references (use no-reference metrics only)\n")
        f.write(f"   Total images to process: {total_images}\n")
        f.write("\n")

        # Estimate processing time
        time_per_image_gpu = 0.12  # seconds (estimated)
        time_per_image_cpu = 0.48  # seconds (estimated)

        total_time_gpu = total_images * time_per_image_gpu / 60  # minutes
        total_time_cpu = total_images * time_per_image_cpu / 60  # minutes

        f.write(" Estimated Processing Time (per ablation):\n")
        f.write(f"   GPU (RTX 3090): ~{total_time_gpu:.1f} minutes\n")
        f.write(f"   CPU: ~{total_time_cpu:.1f} minutes\n")
        f.write("\n")

        # Estimate storage
        storage_per_image = 0.04  # GB (estimated for enhanced images)
        total_storage = total_images * storage_per_image
        total_storage_all = total_storage * 16  # 16 ablations

        f.write(" Estimated Storage Requirements:\n")
        f.write(f"   Per ablation: ~{total_storage:.1f} GB\n")
        f.write(f"   All 16 ablations: ~{total_storage_all:.1f} GB\n")
        f.write("\n")

    logger.info(f"Report saved: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Scan datasets and generate experiment configuration'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='../../data/stage3_eval',
        help='Base directory containing all datasets'
    )
    parser.add_argument(
        '--output-config',
        type=str,
        default='../../configs/experiment_auto.yaml',
        help='Output configuration file path'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default='dataset_scan_report.txt',
        help='Output report file path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(debug=args.debug)

    # Scan datasets
    datasets = scan_all_datasets(args.base_dir, logger)

    if not datasets:
        logger.error("No datasets found!")
        return

    # Generate config file
    generate_config_file(datasets, args.output_config, logger)

    # Generate report
    generate_report(datasets, args.output_report, logger)

    logger.info("")
    logger.info("=" * 100)
    logger.info("Dataset scanning completed successfully!")
    logger.info(f"Found {len(datasets)} datasets with {sum(ds['low_count'] for ds in datasets)} total images")
    logger.info("=" * 100)


if __name__ == '__main__':
    main()
