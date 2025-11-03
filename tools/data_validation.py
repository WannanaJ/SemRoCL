# data_validation.py
import os
import cv2
import numpy as np

def validate_data_structure():
    print("=== 数据集结构验证 ===")
    
    expected_dirs = [
        "data/LOL-v1",
        "data/LOL-v2", 
        "data/LIME"
    ]
    
    for data_dir in expected_dirs:
        if os.path.exists(data_dir):
            print(f"✅ {data_dir} 存在")
            # 检查子目录
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if subdirs:
                print(f"   子目录: {subdirs[:5]}...")  # 显示前5个子目录
            else:
                print(f"   ⚠️ 无子目录或为空")
        else:
            print(f"❌ {data_dir} 不存在")
    
    # 检查是否有图像文件
    print("\n=== 图像文件检查 ===")
    for data_dir in expected_dirs:
        if os.path.exists(data_dir):
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend([f for f in os.listdir(data_dir) if f.lower().endswith(ext[1:])])
            
            if image_files:
                print(f"✅ {data_dir} 包含 {len(image_files)} 个图像文件")
                print(f"   示例: {image_files[:3]}")  # 显示前3个文件
            else:
                print(f"⚠️ {data_dir} 无图像文件")

if __name__ == "__main__":
    validate_data_structure()