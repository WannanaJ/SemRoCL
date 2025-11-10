"""
Utility Functions for SemRoCL
"""


import torch
import numpy as np
import cv2
import os
import yaml
from PIL import Image
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load YAML configuration file with UTF-8 encoding"""
    with open(config_path, 'r', encoding='utf-8') as f:  # ✅ Added encoding
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """Save configuration to YAML file with UTF-8 encoding"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:  # ✅ Added encoding
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return model, optimizer, epoch


def tensor_to_image(tensor, apply_denoise=True, denoise_strength=3):
    """
    Convert tensor to numpy image with optional denoising

    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        apply_denoise: Whether to apply bilateral filter for denoising (default: True)
        denoise_strength: Denoising strength for bilateral filter (default: 3)

    Returns:
        Numpy image array (H, W, C) in uint8 format
    """
    image = tensor.cpu().detach().numpy()
    if image.ndim == 4:
        image = image[0]  # Take first image in batch
    image = np.transpose(image, (1, 2, 0))  # CHW to HWC

    # 使用软裁剪而非硬裁剪，减少色彩断层伪影
    # Soft clipping reduces color banding artifacts
    image = np.clip(image, 0, 1)

    # 转换到 uint8 之前进行降噪处理
    # Apply denoising before uint8 conversion to reduce noise artifacts
    if apply_denoise and denoise_strength > 0:
        # 先转换到 uint8
        image_uint8 = (image * 255).astype(np.uint8)

        # 应用双边滤波器，保留边缘同时去除噪点
        # Bilateral filter preserves edges while removing noise
        image_uint8 = cv2.bilateralFilter(
            image_uint8,
            d=5,  # 滤波器直径 / Filter diameter
            sigmaColor=denoise_strength * 10,  # 色彩空间标准差 / Color space sigma
            sigmaSpace=denoise_strength * 10   # 坐标空间标准差 / Coordinate space sigma
        )
        return image_uint8
    else:
        image = (image * 255).astype(np.uint8)
        return image


def save_image(tensor, filename, quality=95, apply_post_processing=True):
    """
    Save tensor as high-quality image file with optional post-processing

    Args:
        tensor: Input tensor to save
        filename: Output file path
        quality: JPEG/PNG quality (1-100, default: 95)
        apply_post_processing: Apply denoising and enhancement (default: True)
    """
    # 转换为图像，应用降噪
    # Convert to image with denoising
    image = tensor_to_image(tensor, apply_denoise=apply_post_processing, denoise_strength=3)

    # 额外的后处理步骤，减少伪影
    # Additional post-processing to reduce artifacts
    if apply_post_processing:
        # 应用轻微的非局部均值去噪，进一步减少噪点
        # Apply Non-local Means Denoising for further noise reduction
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=5,              # 滤波强度 / Filter strength
            hColor=5,         # 色彩滤波强度 / Color filter strength
            templateWindowSize=7,  # 模板窗口大小 / Template window size
            searchWindowSize=21    # 搜索窗口大小 / Search window size
        )

        # 应用轻微的锐化，补偿降噪造成的模糊
        # Apply subtle sharpening to compensate for denoising blur
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(image, -1, kernel * 0.15)  # 降低锐化强度 / Reduce sharpening intensity
        image = cv2.addWeighted(image, 0.85, sharpened, 0.15, 0)  # 混合原图和锐化图 / Blend original and sharpened

    # 确保目录存在
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    # 使用高质量保存设置
    # Save with high quality settings
    pil_image = Image.fromarray(image)

    # 根据文件扩展名选择保存参数
    # Choose save parameters based on file extension
    if filename.lower().endswith('.png'):
        # PNG 使用最小压缩以保持质量
        # PNG: use minimal compression for quality
        pil_image.save(filename, 'PNG', compress_level=1, optimize=False)
    elif filename.lower().endswith(('.jpg', '.jpeg')):
        # JPEG 使用高质量和优化
        # JPEG: use high quality with optimization
        pil_image.save(filename, 'JPEG', quality=quality, optimize=True, subsampling=0)
    else:
        # 默认保存
        # Default save
        pil_image.save(filename, quality=quality)


def visualize_batch(low_imgs, enhanced_imgs, high_imgs=None, save_path=None):
    """Visualize a batch of images"""
    batch_size = low_imgs.shape[0]
    n_cols = 3 if high_imgs is not None else 2
    
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(n_cols*4, batch_size*4))
    
    for i in range(batch_size):
        # Low-light image
        axes[i, 0].imshow(tensor_to_image(low_imgs[i]))
        axes[i, 0].set_title('Low-light Input')
        axes[i, 0].axis('off')
        
        # Enhanced image
        axes[i, 1].imshow(tensor_to_image(enhanced_imgs[i]))
        axes[i, 1].set_title('Enhanced')
        axes[i, 1].axis('off')
        
        # Ground truth (if available)
        if high_imgs is not None:
            axes[i, 2].imshow(tensor_to_image(high_imgs[i]))
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_exp_dir(base_dir, exp_name):
    """Create experiment directory structure"""
    exp_dir = os.path.join(base_dir, exp_name)
    
    dirs = {
        'checkpoints': os.path.join(exp_dir, 'checkpoints'),
        'logs': os.path.join(exp_dir, 'logs'),
        'results': os.path.join(exp_dir, 'results'),
        'visualizations': os.path.join(exp_dir, 'visualizations')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epochs, lr_decay_rate):
    """Decay learning rate by a factor every lr_decay_epochs"""
    lr = initial_lr
    for milestone in lr_decay_epochs:
        if epoch >= milestone:
            lr *= lr_decay_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def freeze_model(model):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    # Test utilities0
    print("Testing utility functions...")
    
    # Test tensor to image conversion
    test_tensor = torch.randn(1, 3, 256, 256)
    test_image = tensor_to_image(test_tensor)
    print(f"Converted tensor shape {test_tensor.shape} to image shape {test_image.shape}")
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average meter: {meter.avg}")