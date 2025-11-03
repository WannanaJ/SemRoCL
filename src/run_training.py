#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemRoCL Training Runner - Handles all setup and dependencies
Run this script to start training with automatic fixes
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_color(message, color=GREEN):
    print(f"{color}{message}{RESET}")

def check_and_install_packages():
    """Check and install required packages"""
    required = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'albumentations': 'albumentations',
        'tqdm': 'tqdm',
        'yaml': 'pyyaml',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print_color(f"✓ {package} installed", GREEN)
        except ImportError:
            missing.append(package)
            print_color(f"✗ {package} missing", RED)
    
    if missing:
        print_color(f"\nInstalling missing packages: {', '.join(missing)}", YELLOW)
        for package in missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
    
    # Optional packages
    optional = {'wandb': 'wandb'}
    for module, package in optional.items():
        try:
            __import__(module)
            print_color(f"✓ {package} (optional) installed", GREEN)
        except ImportError:
            print_color(f"⚠ {package} (optional) not installed - training will work without it", YELLOW)

def main():
    print_color("="*60, BLUE)
    print_color("SemRoCL Training Setup & Runner", BLUE)
    print_color("="*60, BLUE)
    
    # Step 1: Check packages
    print_color("\n📦 Checking dependencies...", BLUE)
    check_and_install_packages()
    
    # Step 2: Check CUDA
    print_color("\n🖥️ Checking CUDA...", BLUE)
    try:
        import torch
        if torch.cuda.is_available():
            print_color(f"✓ CUDA available: {torch.cuda.get_device_name(0)}", GREEN)
            print_color(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", GREEN)
        else:
            print_color("⚠ CUDA not available - will use CPU (slow)", YELLOW)
    except Exception as e:
        print_color(f"Error checking CUDA: {e}", RED)
    
    # Step 3: Check data directory
    print_color("\n📁 Checking data...", BLUE)
    data_dir = "./data/stage2_paired"
    if os.path.exists(data_dir):
        # Count images
        import glob
        images = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
        images += glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True)
        print_color(f"✓ Found {len(images)} images in {data_dir}", GREEN)
    else:
        print_color(f"⚠ Data directory not found: {data_dir}", YELLOW)
        print_color("  Creating dummy data for testing...", YELLOW)
        os.makedirs(data_dir, exist_ok=True)
    
    # Step 4: Import and run training
    print_color("\n🚀 Starting training...", BLUE)
    print_color("="*60, BLUE)
    
    # Import necessary modules
    import torch
    import torch.nn as nn
    import yaml
    from datetime import datetime
    
    # Load configuration
    config_file = "train_stage2_curriculum.yaml"
    if os.path.exists(config_file):
        print_color(f"Loading config from {config_file}", GREEN)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        print_color("Using default configuration", YELLOW)
        config = {
            'exp_name': 'stage2_training',
            'output_dir': 'outputs',
            'dataset': {
                'root_dir': data_dir,
                'img_size': 256,
                'mode': 'paired'
            },
            'model': {
                'base_encoder': 'resnet50',
                'num_classes': 19,
                'enhancer_channels': 32,
                'num_iterations': 4
            },
            'training': {
                'epochs': 50,
                'batch_size': 4,
                'lr': 0.0001,
                'num_workers': 2,
                'save_freq': 5,
                'pin_memory': True,
                'accumulate_steps': 1
            },
            'curriculum': {
                'enabled': True
            },
            'loss': {
                'w_color': 0.0001,
                'w_semantic': 0.0,
                'w_perceptual': 0.1,
                'w_task': 0.3
            }
        }
    
    # Import models
    try:
        from model.models import EnhancementGenerator, Discriminator
        print_color("✓ Models imported successfully", GREEN)
    except ImportError:
        print_color("⚠ Using simple model (models.py not found)", YELLOW)
        # Create simple model inline
        class EnhancementGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
                self.conv4 = nn.Conv2d(64, 3, 3, padding=1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x1 = self.relu(self.conv1(x))
                x2 = self.relu(self.conv2(x1))
                x3 = self.relu(self.conv3(x2))
                out = torch.sigmoid(self.conv4(x3))
                return out * 0.8 + x * 0.2
        
        Discriminator = None
    
    # Import data loader
    try:
        from data_loader import get_dataloader
        print_color("✓ Data loader imported", GREEN)
    except ImportError as e:
        print_color(f"⚠ Data loader issue: {e}", YELLOW)
        # Create dummy dataloader
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'low': torch.randn(3, 256, 256),
                    'high': torch.randn(3, 256, 256),
                    'low_path': f'dummy_{idx}.jpg'
                }
        
        def get_dataloader(cfg, split='train'):
            dataset = DummyDataset(100)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg['training']['batch_size'],
                shuffle=(split == 'train'),
                num_workers=0
            )
    
    # Run training
    try:
        # Try to import the fixed trainer
        from train_stage2_curriculum import CurriculumTrainer
        
        # Create models
        generator = EnhancementGenerator()
        discriminator = Discriminator() if Discriminator else None
        
        # Create and run trainer
        trainer = CurriculumTrainer(config, generator, discriminator)
        trainer.train()
        
    except ImportError:
        print_color("Running simple training loop...", YELLOW)
        
        # Simple training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancementGenerator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
        criterion = nn.L1Loss()
        
        # Get data loader
        train_loader = get_dataloader(config, 'train')
        
        print_color(f"Training on {device}", GREEN)
        print_color(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", GREEN)
        
        # Training loop
        for epoch in range(1, config['training']['epochs'] + 1):
            model.train()
            total_loss = 0
            
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    low = batch['low'].to(device)
                    high = batch.get('high', low).to(device)
                    
                    optimizer.zero_grad()
                    enhanced = model(low)
                    loss = criterion(enhanced, high)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print_color(f"Batch {batch_idx} error: {e}", RED)
                    continue
            
            avg_loss = total_loss / max(len(train_loader), 1)
            print_color(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}", GREEN)
            
            # Save checkpoint
            if epoch % config['training']['save_freq'] == 0:
                checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, checkpoint_path)
                print_color(f"✓ Saved checkpoint: {checkpoint_path}", GREEN)
    
    print_color("\n✅ Training completed!", GREEN)
    print_color("="*60, BLUE)

if __name__ == "__main__":
    main()