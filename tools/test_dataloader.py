# -*- coding: utf-8 -*-
"""
test_dataloader.py
测试 SemRoCL 项目的数据加载模块
✅ 自动读取 YAML 配置
✅ 自动识别数据集（LOL-v1、LOL-v2、FiveK、Real-LOLBLue、LSRW）
✅ 验证 DataLoader 是否正常工作
"""

import os
import sys
import yaml
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from src.data_loader import get_dataloader
def main():
    print("🔍 正在加载配置文件 train_stage2.yaml ...")
    yaml_path = os.path.join("configs", "train_stage2.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ 找不到配置文件: {yaml_path}")

    # Step 1: 读取配置
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("✅ YAML 配置加载成功：")
    print(cfg)

    # Step 2: 初始化数据加载器
    loader = get_dataloader(cfg, split="train")

    # Step 3: 读取一个 batch 检查
    print("🚀 DataLoader 已构建，测试加载一批数据...")
    try:
        batch = next(iter(loader))
        low = batch["low"]
        high = batch.get("high", None)

        print(f"✅ 成功加载 batch:")
        print(f"   low : {tuple(low.shape)}")
        if high is not None:
            print(f"   high: {tuple(high.shape)}")
        else:
            print("   high: None (unpaired dataset)")

        print(f"🧾 数据类型: low={low.dtype}, device={low.device}")

    except StopIteration:
        print("⚠️ DataLoader 没有返回任何 batch，请检查数据路径是否包含有效图像。")

    except Exception as e:
        print(f"❌ 运行 DataLoader 时出错: {e}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # ✅ 防止多进程冲突
    # ⚠️ Windows 多进程加载器必须放在此保护下！
    main()
