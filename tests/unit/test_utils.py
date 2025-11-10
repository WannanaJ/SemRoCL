"""
Unit Tests for Utility Functions
=================================

Tests for src/utils.py
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils import tensor_to_image, save_image, load_config, AverageMeter


class TestTensorToImage:
    """Test tensor_to_image function"""

    def test_basic_conversion(self, sample_image_tensor):
        """测试基本的tensor到图像转换 / Test basic tensor to image conversion"""
        image = tensor_to_image(sample_image_tensor, apply_denoise=False)

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.shape == (256, 256, 3)  # HWC format
        assert image.min() >= 0 and image.max() <= 255

    def test_with_denoising(self, sample_image_tensor):
        """测试带降噪的转换 / Test conversion with denoising"""
        image = tensor_to_image(sample_image_tensor, apply_denoise=True, denoise_strength=3)

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.shape == (256, 256, 3)

    def test_batch_tensor(self):
        """测试批量tensor转换 / Test batch tensor conversion"""
        batch_tensor = torch.randn(4, 3, 128, 128)
        image = tensor_to_image(batch_tensor, apply_denoise=False)

        # 应该只取第一张 / Should take only first image
        assert image.shape == (128, 128, 3)

    def test_value_range(self, sample_image_tensor):
        """测试值范围裁剪 / Test value range clipping"""
        # 创建超出范围的tensor / Create tensor with values out of range
        tensor = torch.randn(1, 3, 64, 64) * 10  # Values far from [0, 1]
        image = tensor_to_image(tensor, apply_denoise=False)

        assert image.min() >= 0
        assert image.max() <= 255


class TestSaveImage:
    """Test save_image function"""

    def test_save_png(self, sample_image_tensor, temp_output_dir):
        """测试PNG格式保存 / Test PNG format saving"""
        save_path = temp_output_dir / "test.png"
        save_image(sample_image_tensor, str(save_path), apply_post_processing=False)

        assert save_path.exists()
        assert save_path.suffix == '.png'

    def test_save_jpeg(self, sample_image_tensor, temp_output_dir):
        """测试JPEG格式保存 / Test JPEG format saving"""
        save_path = temp_output_dir / "test.jpg"
        save_image(sample_image_tensor, str(save_path), quality=95, apply_post_processing=False)

        assert save_path.exists()
        assert save_path.suffix == '.jpg'

    def test_with_post_processing(self, sample_image_tensor, temp_output_dir):
        """测试带后处理的保存 / Test saving with post-processing"""
        save_path = temp_output_dir / "test_processed.png"
        save_image(sample_image_tensor, str(save_path), apply_post_processing=True)

        assert save_path.exists()

    def test_directory_creation(self, sample_image_tensor, temp_output_dir):
        """测试自动创建目录 / Test automatic directory creation"""
        nested_path = temp_output_dir / "subdir" / "nested" / "test.png"
        save_image(sample_image_tensor, str(nested_path), apply_post_processing=False)

        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestAverageMeter:
    """Test AverageMeter class"""

    def test_initialization(self):
        """测试初始化 / Test initialization"""
        meter = AverageMeter()

        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_single_update(self):
        """测试单次更新 / Test single update"""
        meter = AverageMeter()
        meter.update(10)

        assert meter.val == 10
        assert meter.avg == 10
        assert meter.count == 1

    def test_multiple_updates(self):
        """测试多次更新 / Test multiple updates"""
        meter = AverageMeter()
        values = [10, 20, 30, 40, 50]

        for val in values:
            meter.update(val)

        assert meter.avg == 30.0
        assert meter.count == 5

    def test_weighted_update(self):
        """测试加权更新 / Test weighted update"""
        meter = AverageMeter()
        meter.update(10, n=2)
        meter.update(20, n=3)

        expected_avg = (10 * 2 + 20 * 3) / 5
        assert meter.avg == expected_avg

    def test_reset(self):
        """测试重置功能 / Test reset functionality"""
        meter = AverageMeter()
        meter.update(100)
        meter.reset()

        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0


class TestConfigLoading:
    """Test configuration loading"""

    def test_load_valid_config(self, tmp_path):
        """测试加载有效配置 / Test loading valid config"""
        import yaml

        config_data = {
            'exp_name': 'test',
            'model': {'channels': 32},
            'training': {'epochs': 100}
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

        loaded_config = load_config(str(config_file))

        assert loaded_config == config_data
        assert loaded_config['exp_name'] == 'test'
