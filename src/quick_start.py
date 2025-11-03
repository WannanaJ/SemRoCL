#!/usr/bin/env python3
"""
SemRoCL Quick Start Script
Sets up and runs training with proper error handling
"""

import os
import sys
import json
import torch
import torch.nn as nn
from datetime import datetime


def create_simple_model():
    """Create a simple but effective enhancement model"""
    
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
        
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)
    
    class SimpleEnhancementNet(nn.Module):
        def __init__(self, num_blocks=4):
            super().__init__()
            
            # Initial convolution
            self.conv_in = nn.Conv2d(3, 64, 7, padding=3)
            
            # Residual blocks
            self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
            
            # Output convolution
            self.conv_out = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Remember input for residual connection
            input_img = x
            
            # Process through network
            x = self.conv_in(x)
            x = self.res_blocks(x)
            x = self.conv_out(x)
            
            # Add residual connection for better gradient flow
            return x * 0.8 + input_img * 0.2
    
    return SimpleEnhancementNet()


def create_training_config():
    """Create a working training configuration"""
    
    config = {
        "model": {
            "name": "SimpleEnhancementNet",
            "num_blocks": 4
        },
        "dataset": {
            "root_dir": "./data/stage2_paired",
            "mode": "paired",
            "img_size": 256,  # Start small for faster iteration
            "augment": True,
            "normalize": True,
            "cache_images": False,
            "use_pil": True
        },
        "training": {
            "epochs": 100,
            "batch_size": 4,  # Small batch size for stability
            "num_workers": 2,  # Fewer workers to avoid issues
            "mixed_precision": False,  # Disable for debugging
            "save_interval": 5,
            "checkpoint_dir": "./checkpoints",
            "optimizer": {
                "type": "adam",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 1e-5
            },
            "scheduler": {
                "type": "step",
                "step_size": 30,
                "gamma": 0.5
            },
            "gradient_clip": 1.0,
            "persistent_workers": False,  # Disable to avoid worker issues
            "prefetch_factor": 2
        },
        "losses": {
            "use_perceptual": False,  # Start simple
            "use_contrastive": False,
            "use_ssim": False,
            "use_illumination": False,
            "weights": {
                "l1": 1.0  # Start with L1 loss only
            }
        },
        "logging": {
            "use_wandb": False,
            "log_interval": 10,
            "sample_interval": 100
        }
    }
    
    return config


def safe_train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Safe training loop with comprehensive error handling"""
    
    model.train()
    total_loss = 0
    num_batches = 0
    l1_loss = nn.L1Loss()
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Safely get batch data
            if "low" not in batch:
                print(f"Warning: 'low' not in batch at index {batch_idx}")
                continue
            
            low_img = batch["low"].to(device)
            
            # Check if we have ground truth
            if "high" in batch and batch["high"] is not None:
                high_img = batch["high"].to(device)
            else:
                # Self-supervised: use input as target
                high_img = low_img
            
            # Forward pass
            optimizer.zero_grad()
            enhanced = model(low_img)
            
            # Compute loss
            loss = l1_loss(enhanced, high_img)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg": f"{total_loss/max(num_batches, 1):.4f}"
            })
            
        except Exception as e:
            print(f"\n⚠️ Error in batch {batch_idx}: {str(e)}")
            print(f"   Batch keys: {batch.keys() if batch else 'None'}")
            if "low" in batch:
                print(f"   Low shape: {batch['low'].shape}")
            if "high" in batch:
                print(f"   High shape: {batch['high'].shape if batch['high'] is not None else 'None'}")
            
            # Continue training despite error
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def quick_start_training():
    """Quick start training with minimal dependencies"""
    
    print("🚀 SemRoCL Quick Start Training")
    print("=" * 60)
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create configuration
    config = create_training_config()
    print(f"\n📋 Configuration:")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Image Size: {config['dataset']['img_size']}")
    print(f"   Learning Rate: {config['training']['optimizer']['lr']}")
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_simple_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        betas=tuple(config['training']['optimizer']['betas']),
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler']['step_size'],
        gamma=config['training']['scheduler']['gamma']
    )
    
    # Try to create data loader
    print("\n📁 Setting up data loader...")
    try:
        from data_loader import get_dataloader
        train_loader = get_dataloader(config, split="train")
        print(f"   Train batches: {len(train_loader)}")
    except Exception as e:
        print(f"⚠️ Could not create data loader: {e}")
        print("   Creating dummy data loader for testing...")
        
        # Create dummy dataset for testing
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    "low": torch.randn(3, 256, 256),
                    "high": torch.randn(3, 256, 256),
                    "low_path": f"dummy_{idx}.jpg"
                }
        
        train_loader = torch.utils.data.DataLoader(
            DummyDataset(100),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        print(f"   Using dummy dataset with {len(train_loader)} batches")
    
    # Training loop
    print("\n🏃 Starting training...")
    print("-" * 60)
    
    best_loss = float('inf')
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train one epoch
        avg_loss = safe_train_epoch(
            model, train_loader, optimizer, device, 
            epoch, config['training']['epochs']
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📊 Epoch {epoch}/{config['training']['epochs']}")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"   ✅ Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'config': config
                }, best_path)
                print(f"   ⭐ New best model saved! Loss: {best_loss:.4f}")
        
        print("-" * 60)
    
    print("\n✅ Training completed!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints saved in: {checkpoint_dir}")
    
    # Save final config
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved: {config_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SemRoCL Quick Start')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Run diagnostics instead of training')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to data directory')
    args = parser.parse_args()
    
    if args.diagnose:
        # Run diagnostics
        os.system("python diagnose_training.py")
    else:
        # Check for required packages
        required_packages = ['torch', 'torchvision', 'tqdm', 'numpy', 'pillow']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"⚠️ Missing packages: {', '.join(missing)}")
            print(f"   Install with: pip install {' '.join(missing)}")
            print("\nOr install all dependencies:")
            print("   pip install torch torchvision tqdm numpy pillow opencv-python albumentations")
            sys.exit(1)
        
        # Run training
        quick_start_training()


if __name__ == "__main__":
    main()