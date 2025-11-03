#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键修复脚本 - 自动修复 KeyError: 'lr' 问题

使用方法：
  python fix_config.py train_stage2.yaml

功能：
  - 自动在配置文件中添加 'lr' 字段
  - 保持所有优化配置不变
  - 创建备份
"""

import yaml
import sys
import os
import shutil
from datetime import datetime

def fix_config(config_path):
    """修复配置文件"""
    
    # 创建备份
    backup_path = config_path + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2(config_path, backup_path)
    print(f"✓ 创建备份: {backup_path}")
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否需要修复
    if 'lr' in config.get('training', {}):
        print("✓ 配置文件已包含 'lr' 字段，无需修复")
        return
    
    # 添加 lr 字段
    if 'training' not in config:
        config['training'] = {}
    
    # 使用 lr_generator 的值，或默认 0.0001
    lr_value = config['training'].get('lr_generator', 0.0001)
    
    # 保存原有的 training 字典顺序
    training = config['training']
    new_training = {}
    
    # 首先添加基本字段
    for key in ['epochs', 'batch_size']:
        if key in training:
            new_training[key] = training[key]
    
    # 添加 lr 字段（兼容性）
    new_training['lr'] = lr_value
    new_training['_lr_note'] = '(for backward compatibility)'
    
    # 添加其余字段
    for key, value in training.items():
        if key not in new_training:
            new_training[key] = value
    
    config['training'] = new_training
    
    # 写回文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ 修复完成: {config_path}")
    print(f"  添加了 'lr: {lr_value}' 字段")
    print(f"\n现在可以运行: python train_stage2_enhance.py --config {config_path}")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python fix_config.py <config_file.yaml>")
        print("\n示例:")
        print("  python fix_config.py train_stage2.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"❌ 错误: 文件不存在: {config_path}")
        sys.exit(1)
    
    if not config_path.endswith(('.yaml', '.yml')):
        print(f"⚠️  警告: 文件扩展名不是 .yaml 或 .yml")
        response = input("继续处理？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            sys.exit(0)
    
    try:
        fix_config(config_path)
        print("\n✅ 修复成功！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print(f"   请检查配置文件格式是否正确")
        sys.exit(1)

if __name__ == '__main__':
    main()