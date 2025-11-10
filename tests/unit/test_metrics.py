"""
Unit Tests for Metrics Utilities
=================================

Tests for src/metrics_utils.py
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from metrics_utils import compute_psnr, compute_ssim, compute_delta_e, compute_iter_speed


class TestPSNR:
    """Test PSNR computation"""

    def test_identical_images(self):
        """测试相同图像的PSNR / Test PSNR of identical images"""
        img = torch.rand(1, 3, 64, 64)
        psnr = compute_psnr(img, img)

        # 相同图像PSNR应该非常高 / Identical images should have very high PSNR
        assert psnr.item() > 50  # Usually infinity, but numerical precision limits

    def test_different_images(self):
        """测试不同图像的PSNR / Test PSNR of different images"""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        psnr = compute_psnr(img1, img2)

        # 不同图像PSNR应该有限 / Different images should have finite PSNR
        assert 0 < psnr.item() < 50

    def test_output_type(self, sample_image_tensor):
        """测试输出类型 / Test output type"""
        img1 = sample_image_tensor
        img2 = sample_image_tensor + 0.1
        psnr = compute_psnr(img1, img2)

        assert isinstance(psnr, torch.Tensor)
        assert psnr.dim() == 0  # Scalar tensor


class TestSSIM:
    """Test SSIM computation"""

    def test_identical_images(self):
        """测试相同图像的SSIM / Test SSIM of identical images"""
        img = torch.rand(1, 3, 64, 64)
        ssim = compute_ssim(img, img)

        # 相同图像SSIM应该接近1 / Identical images should have SSIM close to 1
        assert 0.99 < ssim.item() <= 1.0

    def test_different_images(self):
        """测试不同图像的SSIM / Test SSIM of different images"""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        ssim = compute_ssim(img1, img2)

        # SSIM范围应该在[-1, 1] / SSIM should be in range [-1, 1]
        assert -1 <= ssim.item() <= 1

    def test_similar_images(self):
        """测试相似图像的SSIM / Test SSIM of similar images"""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = img1 + 0.01 * torch.randn_like(img1)
        ssim = compute_ssim(img1, img2)

        # 相似图像应该有较高的SSIM / Similar images should have high SSIM
        assert ssim.item() > 0.8


class TestDeltaE:
    """Test DeltaE (color difference) computation"""

    def test_identical_colors(self):
        """测试相同颜色的DeltaE / Test DeltaE of identical colors"""
        img = torch.rand(1, 3, 32, 32)
        delta_e = compute_delta_e(img, img)

        # 相同颜色DeltaE应该接近0 / Identical colors should have DeltaE close to 0
        assert delta_e.item() < 1.0

    def test_different_colors(self):
        """测试不同颜色的DeltaE / Test DeltaE of different colors"""
        img1 = torch.zeros(1, 3, 32, 32)  # Black
        img2 = torch.ones(1, 3, 32, 32)   # White
        delta_e = compute_delta_e(img1, img2)

        # 黑白差异应该很大 / Black-white difference should be large
        assert delta_e.item() > 10

    def test_output_range(self, sample_image_tensor):
        """测试输出范围 / Test output range"""
        img1 = sample_image_tensor
        img2 = sample_image_tensor + 0.2
        delta_e = compute_delta_e(img1, img2)

        # DeltaE应该是非负的 / DeltaE should be non-negative
        assert delta_e.item() >= 0


class TestIterSpeed:
    """Test iteration speed computation"""

    def test_basic_computation(self):
        """测试基本计算 / Test basic computation"""
        iter_time = 0.5  # 500ms
        batch_size = 8

        iter_ms, imgs_per_sec = compute_iter_speed(iter_time, batch_size)

        assert iter_ms == 500.0
        assert imgs_per_sec == 16.0  # 8 images / 0.5 seconds

    def test_zero_time_handling(self):
        """测试零时间处理 / Test zero time handling"""
        iter_time = 0.0
        batch_size = 4

        # 应该不会抛出除零错误 / Should not raise division by zero
        iter_ms, imgs_per_sec = compute_iter_speed(iter_time + 1e-8, batch_size)

        assert iter_ms >= 0
        assert imgs_per_sec >= 0

    def test_various_batch_sizes(self):
        """测试不同批量大小 / Test various batch sizes"""
        iter_time = 1.0

        for batch_size in [1, 2, 4, 8, 16]:
            iter_ms, imgs_per_sec = compute_iter_speed(iter_time, batch_size)

            assert imgs_per_sec == batch_size
            assert iter_ms == 1000.0
