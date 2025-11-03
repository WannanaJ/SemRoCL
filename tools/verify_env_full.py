# ===============================================================
# verify_env_full.py
# 一键检测深度学习环境（PyTorch + CUDA + 图像库）
# 适配环境: semrocl (PyTorch 2.5.1 + CUDA 12.1)
# ===============================================================

import os
import torch
import torchvision
from PIL import Image
import cv2
import platform
import sys

print("=== 环境全面检测 ===\n")

# 1️⃣ 系统与 Python 信息
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python路径: {sys.executable}")
print(f"Python版本: {platform.python_version()}\n")

# 2️⃣ PyTorch 检查
print("【PyTorch】")
print(f"版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前设备索引: {torch.cuda.current_device()}")
else:
    print("⚠️ CUDA 未启用或驱动不匹配。")

# 3️⃣ TorchVision 检查
print("\n【TorchVision】")
print(f"版本: {torchvision.__version__}")
try:
    from torchvision import models
    _ = models.resnet18(weights=None)
    print("✅ TorchVision 模型加载测试通过")
except Exception as e:
    print("❌ TorchVision 加载错误:", e)

# 4️⃣ Pillow 检查
print("\n【Pillow】")
try:
    img = Image.new("RGB", (128, 128), color="red")
    img.save("verify_test_image.png")
    print("✅ Pillow 正常，可创建图像 verify_test_image.png")
except Exception as e:
    print("❌ Pillow 错误:", e)

# 5️⃣ OpenCV 检查
print("\n【OpenCV】")
try:
    print("版本:", cv2.__version__)
    test_img = cv2.imread("verify_test_image.png")
    if test_img is not None:
        cv2.imwrite("verify_test_image_cv2.png", test_img)
        print("✅ OpenCV 正常，可读写图像 verify_test_image_cv2.png")
    else:
        print("⚠️ OpenCV 读取测试图片失败。")
except Exception as e:
    print("❌ OpenCV 错误:", e)

# 6️⃣ 额外验证: GPU张量运算
print("\n【GPU Tensor 测试】")
try:
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device='cuda')
        y = torch.mm(x, x)
        print("✅ GPU矩阵运算成功:", y.shape)
    else:
        print("⚠️ 跳过: 当前无CUDA可用。")
except Exception as e:
    print("❌ GPU 计算错误:", e)

print("\n=== 检测完成 ===")
