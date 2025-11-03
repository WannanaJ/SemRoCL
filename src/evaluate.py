"""
Evaluation Script for SemRoCL
Computes PSNR, SSIM, LPIPS, NIQE metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from src.data_loader import LowLightDataset
from model.enhancer import CurveEnhancer
from model.semantic_head import SemanticHead
from utils import *


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1_np = tensor_to_image(img1) / 255.0
    img2_np = tensor_to_image(img2) / 255.0
    return psnr(img1_np, img2_np, data_range=1.0)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    img1_np = tensor_to_image(img1) / 255.0
    img2_np = tensor_to_image(img2) / 255.0
    return ssim(img1_np, img2_np, multichannel=True, data_range=1.0, channel_axis=2)


def evaluate(config):
    """Evaluate trained model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    enhancer = CurveEnhancer(n_channels=32, num_iterations=8, use_semantic=True).to(device)
    semantic_head = SemanticHead(num_classes=19, backbone='segformer-b0').to(device)
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint'])
    enhancer.load_state_dict(checkpoint['enhancer'])
    semantic_head.load_state_dict(checkpoint['semantic_head'])
    
    enhancer.eval()
    semantic_head.eval()
    
    # Load test data
    test_loader = DataLoader(
        LowLightDataset(config['dataset']['root_dir'], mode='paired', 
                       split='test', img_size=config['dataset']['img_size'], augment=False),
        batch_size=1, shuffle=False, num_workers=2
    )
    
    # LPIPS calculator
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Metrics storage
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    # Evaluation loop
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            low = batch['low'].to(device)
            high = batch['high'].to(device)
            
            # Get semantic features
            seg_logits, confidence = semantic_head(low)
            semantic_features = semantic_head.get_semantic_features(low)[0]
            
            # Enhance
            enhanced, _ = enhancer(low, semantic_features, confidence)
            
            # Calculate metrics
            psnr_val = calculate_psnr(enhanced, high)
            ssim_val = calculate_ssim(enhanced, high)
            
            # LPIPS
            enhanced_norm = enhanced * 2 - 1
            high_norm = high * 2 - 1
            lpips_val = lpips_fn(enhanced_norm, high_norm).item()
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            lpips_scores.append(lpips_val)
            
            # Save result
            if config.get('save_results'):
                filename = batch['filename'][0]
                save_path = os.path.join(config['output_dir'], filename)
                save_image(enhanced, save_path)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    
    return {
        'psnr': np.mean(psnr_scores),
        'ssim': np.mean(ssim_scores),
        'lpips': np.mean(lpips_scores)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to eval config')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['checkpoint'] = args.checkpoint
    
    evaluate(config)
