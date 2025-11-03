import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== 环境测试 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")
print(f"OpenCV版本: {cv2.__version__}")

# 测试 GPU 张量计算
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"GPU矩阵乘法完成: {z.shape}")

print("环境测试完成！")