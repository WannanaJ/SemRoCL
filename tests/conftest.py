"""
Pytest Configuration and Fixtures
==================================

Shared fixtures and configuration for all tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get available device (CPU or CUDA)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image_tensor():
    """Generate a sample image tensor for testing (B, C, H, W)"""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_low_light_tensor():
    """Generate a low-light image tensor (darker)"""
    return torch.randn(1, 3, 256, 256) * 0.3  # 降低亮度 / Reduce brightness


@pytest.fixture
def sample_batch():
    """Generate a batch of images for testing"""
    return {
        'low': torch.randn(4, 3, 256, 256) * 0.3,
        'high': torch.randn(4, 3, 256, 256),
        'filename': ['test_001.png', 'test_002.png', 'test_003.png', 'test_004.png']
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'exp_name': 'test_experiment',
        'output_dir': 'outputs',
        'dataset': {
            'root_dir': './data/test',
            'img_size': 256,
            'mode': 'paired'
        },
        'model': {
            'enhancer_channels': 16,  # 减小用于测试 / Smaller for testing
            'num_iterations': 4,
            'semantic_channels': 32,
            'use_semantic': True,
            'use_multiscale': False
        },
        'training': {
            'epochs': 2,
            'batch_size': 2,
            'lr_generator': 0.0001,
            'lr_discriminator': 0.00001
        },
        'loss': {
            'w_color': 1.0,
            'w_semantic': 0.5,
            'w_perceptual': 1.0,
            'w_freq': 0.1,
            'w_adv': 0.1
        }
    }


@pytest.fixture(autouse=True)
def seed_random():
    """Set random seeds for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    yield
