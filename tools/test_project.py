import sys
import os

print("=== SemRoCL 项目测试 ===")

# 检查关键文件是否存在
required_files = [
    "src/train_stage1_moco.py",
    "src/train_stage2_enhance.py", 
    "src/evaluate.py",
    "configs/train_stage1.yaml",
    "configs/train_stage2.yaml"
]

print("检查关键文件...")
for file in required_files:
    exists = os.path.exists(file)
    status = "✅ 存在" if exists else "❌ 缺失"
    print(f"{status}: {file}")

# 使用正确的类名
print("\n检查模型导入...")
try:
    from src.model.encoder_moco import MoCoV3Encoder
    print("✅ MoCoV3Encoder 导入成功")
except ImportError as e:
    print(f"❌ MoCoV3Encoder 导入失败: {e}")

try:
    from src.model.enhancer import CurveEnhancer
    print("✅ CurveEnhancer 导入成功")
except ImportError as e:
    print(f"❌ CurveEnhancer 导入失败: {e}")

try:
    from src.model.semantic_head import SemanticHead
    print("✅ SemanticHead 导入成功")
except ImportError as e:
    print(f"❌ SemanticHead 导入失败: {e}")

print("\n项目测试完成！")