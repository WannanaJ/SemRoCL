#!/usr/bin/env python3
"""
SemRoCL Training Diagnostics and Fix Script
Identifies and resolves common training issues
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional
import argparse
import traceback


class TrainingDiagnostics:
    """Comprehensive diagnostics for SemRoCL training issues"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def check_cuda_availability(self):
        """Check CUDA and GPU availability"""
        print("\n🔍 Checking CUDA/GPU Setup...")
        
        if not torch.cuda.is_available():
            self.issues_found.append("CUDA not available")
            print(" CUDA is not available. Training will use CPU (very slow)")
            print("   Fix: Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            print(f" CUDA is available")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   Current Device: {torch.cuda.current_device()}")
            print(f"   Device Name: {torch.cuda.get_device_name(0)}")
            
            # Check GPU memory
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"   GPU Memory: {mem_allocated:.2f}/{total_mem:.2f} GB allocated")
            
            if mem_allocated > total_mem * 0.9:
                self.issues_found.append("GPU memory almost full")
                print("  GPU memory is almost full. Consider:")
                print("    - Reducing batch size")
                print("    - Using gradient accumulation")
                print("    - Clearing cache: torch.cuda.empty_cache()")
    
    def check_data_loader_issues(self, data_path: str = None):
        """Check for common data loader problems"""
        print("\n Checking Data Loader...")
        
        issues = []
        
        # Check if data path exists
        if data_path:
            if not os.path.exists(data_path):
                issues.append(f"Data path does not exist: {data_path}")
                print(f"Data path not found: {data_path}")
            else:
                # Check for images
                image_count = 0
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    import glob
                    image_count += len(glob.glob(os.path.join(data_path, f"**/*{ext}"), recursive=True))
                
                if image_count == 0:
                    issues.append("No images found in data directory")
                    print(f" No images found in {data_path}")
                else:
                    print(f" Found {image_count} images")
        
        # Check for albumentations issues
        try:
            import albumentations as A
            print(" Albumentations imported successfully")
            
            # Test basic transform
            test_transform = A.Compose([
                A.RandomCrop(height=256, width=256),
                A.Normalize(mean=(0,0,0), std=(1,1,1))
            ])
            
            # Test with dummy image
            dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            result = test_transform(image=dummy_img)
            print(" Albumentations transforms working")
            
        except Exception as e:
            issues.append(f"Albumentations error: {str(e)}")
            print(f"Albumentations issue: {e}")
            print("   Fix: pip install -U albumentations")
        
        self.issues_found.extend(issues)
        return len(issues) == 0
    
    def check_loss_function_initialization(self):
        """Check if loss functions are properly initialized"""
        print("\n Checking Loss Functions...")
        
        try:
            # Try importing loss functions
            from loss_functions import (
                CurriculumContrastiveLoss,
                AdaptivePerceptualLoss,
                MultiScaleSSIMLoss
            )
            print(" Loss functions imported successfully")
            
            # Test initialization
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Test basic losses
            contrastive = CurriculumContrastiveLoss().to(device)
            print(" CurriculumContrastiveLoss initialized")
            
            perceptual = AdaptivePerceptualLoss().to(device)
            print(" AdaptivePerceptualLoss initialized")
            
            ssim = MultiScaleSSIMLoss().to(device)
            print(" MultiScaleSSIMLoss initialized")
            
            # Test with dummy tensors
            dummy_tensor = torch.randn(2, 3, 256, 256).to(device)
            
            # Test SSIM
            ssim_loss = ssim(dummy_tensor, dummy_tensor)
            assert ssim_loss.item() >= 0, "SSIM loss should be non-negative"
            print(f" SSIM loss test passed: {ssim_loss.item():.4f}")
            
        except ImportError as e:
            self.issues_found.append(f"Loss function import error: {str(e)}")
            print(f" Cannot import loss functions: {e}")
            print("   Make sure loss_functions.py is in the same directory")
        except Exception as e:
            self.issues_found.append(f"Loss function error: {str(e)}")
            print(f" Loss function initialization error: {e}")
    
    def fix_unbound_local_error(self, script_path: str):
        """Fix the UnboundLocalError in training script"""
        print("\n Fixing UnboundLocalError...")
        
        if not os.path.exists(script_path):
            print(f" Script not found: {script_path}")
            return False
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for the problematic pattern
        if 'losses_G["total"].item()' in content and 'losses_G = {}' not in content[:content.find('losses_G["total"].item()')]:
            print(" Found UnboundLocalError pattern")
            
            # The fix is to initialize losses_G at the beginning of the loop
            fix_pattern = """
            # Initialize losses_G at the beginning of each iteration
            losses_G = {
                "total": torch.tensor(0.0, device=self.device),
                "l1": torch.tensor(0.0, device=self.device),
                "perceptual": torch.tensor(0.0, device=self.device),
                "ssim": torch.tensor(0.0, device=self.device),
                "contrastive": torch.tensor(0.0, device=self.device),
                "gan": torch.tensor(0.0, device=self.device)
            }
            """
            
            print(" Fix: Initialize losses_G dictionary at the start of each training iteration")
            print("   Add this code at the beginning of your training loop:")
            print(fix_pattern)
            
            self.fixes_applied.append("UnboundLocalError fix suggested")
            return True
        else:
            print(" No UnboundLocalError pattern detected")
            return False
    
    def check_model_architecture(self):
        """Check if model architecture is properly defined"""
        print("\n Checking Model Architecture...")
        
        try:
            # Try to import models
            from models import EnhancementGenerator
            print(" Model imported successfully")
            
            # Test initialization
            model = EnhancementGenerator()
            print(f" Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
            
            # Test forward pass
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f" Forward pass successful: input {dummy_input.shape} -> output {output.shape}")
            
        except ImportError:
            self.issues_found.append("Model not found")
            print(" Cannot import model. Create a models.py file with EnhancementGenerator class")
            
            # Suggest a simple model
            print("\n Suggested minimal model structure:")
            print("""
```python
# models.py
import torch
import torch.nn as nn

class EnhancementGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
```
            """)
        except Exception as e:
            self.issues_found.append(f"Model error: {str(e)}")
            print(f" Model error: {e}")
    
    def check_training_config(self, config: Dict = None):
        """Validate training configuration"""
        print("\n Checking Training Configuration...")
        
        if not config:
            print("  No configuration provided, using defaults")
            return
        
        # Check critical parameters
        issues = []
        
        # Batch size
        batch_size = config.get("training", {}).get("batch_size", 4)
        if batch_size > 16:
            issues.append(f"Batch size {batch_size} might be too large for most GPUs")
            print(f"  Large batch size: {batch_size}. Consider reducing if OOM errors occur")
        else:
            print(f" Batch size: {batch_size}")
        
        # Learning rate
        lr = config.get("training", {}).get("optimizer", {}).get("lr", 1e-4)
        if lr > 1e-2:
            issues.append(f"Learning rate {lr} might be too high")
            print(f"  High learning rate: {lr}. Consider using 1e-4 to 1e-3")
        else:
            print(f" Learning rate: {lr}")
        
        # Image size
        img_size = config.get("dataset", {}).get("img_size", 512)
        if img_size > 1024:
            issues.append(f"Image size {img_size} is very large, may cause OOM")
            print(f"  Large image size: {img_size}. Consider 256-512 for training")
        else:
            print(f" Image size: {img_size}")
        
        # Workers
        num_workers = config.get("training", {}).get("num_workers", 4)
        if num_workers > os.cpu_count():
            issues.append(f"num_workers {num_workers} exceeds CPU count {os.cpu_count()}")
            print(f"  num_workers ({num_workers}) > CPU count ({os.cpu_count()})")
        else:
            print(f" Number of workers: {num_workers}")
        
        self.issues_found.extend(issues)
    
    def suggest_fixes(self):
        """Provide comprehensive fix suggestions"""
        print("\n" + "="*60)
        print(" DIAGNOSTIC SUMMARY")
        print("="*60)
        
        if not self.issues_found:
            print(" No critical issues found!")
        else:
            print(f"\n  Found {len(self.issues_found)} issues:\n")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
            
            print("\n RECOMMENDED FIXES:")
            print("-" * 40)
            
            # Provide specific fixes for common issues
            if "CUDA not available" in str(self.issues_found):
                print("\n1. Fix CUDA/GPU:")
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            
            if "Data path" in str(self.issues_found):
                print("\n2. Fix Data Path:")
                print("   - Verify your dataset is in the correct directory")
                print("   - Expected structure:")
                print("     stage2_paired/")
                print("     └── LOL-v1/")
                print("         ├── low/")
                print("         └── high/")
            
            if "Loss function" in str(self.issues_found):
                print("\n3. Fix Loss Functions:")
                print("   - Ensure loss_functions.py is in the same directory")
                print("   - Install required packages:")
                print("     pip install torchvision pytorch-msssim")
            
            if "Model not found" in str(self.issues_found):
                print("\n4. Fix Model:")
                print("   - Create models.py with EnhancementGenerator class")
                print("   - Or modify the import in train script to use your model")
        
        if self.fixes_applied:
            print(f"\n Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"   - {fix}")
        
        print("\n" + "="*60)
        print(" Quick Start Commands:")
        print("-" * 40)
        print("1. Install dependencies:")
        print("   pip install torch torchvision albumentations tqdm wandb pillow opencv-python")
        print("\n2. Prepare data structure:")
        print("   mkdir -p data/stage2_paired/LOL-v1/{low,high}")
        print("\n3. Run fixed training script:")
        print("   python train_stage2_curriculum_fixed.py")
        print("="*60)


def main():
    """Run diagnostics"""
    parser = argparse.ArgumentParser(description='SemRoCL Training Diagnostics')
    parser.add_argument('--script', type=str, default='train_stage2_curriculum.py',
                       help='Path to training script to diagnose')
    parser.add_argument('--data', type=str, help='Path to data directory')
    parser.add_argument('--fix', action='store_true', help='Apply automatic fixes')
    args = parser.parse_args()
    
    print(" SemRoCL Training Diagnostics")
    print("=" * 60)
    
    diagnostics = TrainingDiagnostics()
    
    # Run all checks
    diagnostics.check_cuda_availability()
    diagnostics.check_data_loader_issues(args.data)
    diagnostics.check_loss_function_initialization()
    diagnostics.check_model_architecture()
    
    if args.script and os.path.exists(args.script):
        diagnostics.fix_unbound_local_error(args.script)
    
    # Provide summary and fixes
    diagnostics.suggest_fixes()
    
    print("\n Diagnostics complete!")


if __name__ == "__main__":
    main()