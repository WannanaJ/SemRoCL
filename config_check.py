# config_check.py
import yaml
import os

def check_config_files():
    print("=== 配置文件检查 ===")
    
    config_files = [
        "configs/train_stage1.yaml",
        "configs/train_stage2.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} 存在")
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   内容预览: {list(config.keys()) if config else '空文件'}")
                if config:
                    for key, value in config.items():
                        print(f"   {key}: {value}")
            except Exception as e:
                print(f"   ❌ 读取错误: {e}")
        else:
            print(f"❌ {config_file} 不存在")

if __name__ == "__main__":
    check_config_files()