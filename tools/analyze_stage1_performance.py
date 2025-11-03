import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from scipy import stats
import re

# ==============================
# 动态导入 MoCo 编码器
# ==============================
try:
    from encoder_moco import MoCoV3Encoder
    print("✅ Successfully imported MoCoV3Encoder from encoder_moco")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Trying alternative import paths...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from encoder_moco import MoCoV3Encoder
        print("✅ Successfully imported from src directory")
    except ImportError:
        print("💡 Creating a minimal MoCoV3Encoder class for analysis...")
        import torch.nn as nn
        from torchvision import models
        
        class MoCoV3Encoder(nn.Module):
            def __init__(self, base_encoder='resnet18', feat_dim=128):
                super(MoCoV3Encoder, self).__init__()
                if base_encoder == 'resnet18':
                    base_model = models.resnet18(pretrained=False)
                    base_feat_dim = 512
                else:
                    base_model = models.resnet50(pretrained=False)
                    base_feat_dim = 2048
                
                self.encoder = nn.Sequential(
                    *list(base_model.children())[:-1],
                    nn.Flatten(),
                    nn.Linear(base_feat_dim, feat_dim)
                )
            
            def forward(self, x):
                return self.encoder(x)
            
            def get_features(self, x, normalize=True):
                features = self.encoder(x)
                if normalize:
                    features = nn.functional.normalize(features, dim=1)
                return features

# ==============================
# 配置区
# ==============================
CKPT_DIR = "outputs/moco_pretrain_stage1/checkpoints"
DATA_DIR = "data/stage1_unpaired"
PLOT_DIR = "outputs/moco_pretrain_stage1/analysis_plots"
SAMPLE_LIMIT = 200
IMG_SIZE = 224

os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================
# 加载模型
# ==============================
def load_moco_encoder(weight_path):
    print(f"🔍 Loading MoCo encoder from: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu")

    model = MoCoV3Encoder(base_encoder='resnet18', feat_dim=128)
    
    if "encoder_q" in ckpt:
        state_dict = ckpt["encoder_q"]
        print("✅ Loaded state_dict from 'encoder_q'")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("✅ Loaded state_dict from 'model_state_dict'")
    else:
        state_dict = ckpt
        print("⚠️ Using entire checkpoint as state_dict")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ==============================
# 通过checkpoint分析训练趋势
# ==============================
def analyze_training_from_checkpoints():
    """通过分析不同epoch的checkpoint来推断训练趋势"""
    print("📊 Analyzing training trend from checkpoints...")
    
    # 获取所有checkpoint文件
    ckpt_files = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")])
    epoch_ckpts = []
    
    # 提取epoch信息
    for ckpt_file in ckpt_files:
        if "final" in ckpt_file:
            epoch_num = 200  # 假设final是第200个epoch
        else:
            # 从文件名提取epoch数字
            match = re.search(r'epoch_(\d+)', ckpt_file)
            if match:
                epoch_num = int(match.group(1))
            else:
                continue
        epoch_ckpts.append((epoch_num, ckpt_file))
    
    # 按epoch排序
    epoch_ckpts.sort(key=lambda x: x[0])
    
    print(f"📁 Found {len(epoch_ckpts)} checkpoint files from epoch {epoch_ckpts[0][0]} to {epoch_ckpts[-1][0]}")
    
    if len(epoch_ckpts) < 3:
        print("❌ Not enough checkpoints for trend analysis")
        return None
    
    # 选择几个关键epoch进行分析
    sample_epochs = []
    total_epochs = len(epoch_ckpts)
    
    # 选择开始、中间、结束的checkpoint
    indices = [0, total_epochs//4, total_epochs//2, 3*total_epochs//4, -1]
    sample_epochs = [epoch_ckpts[i] for i in indices if i < total_epochs]
    
    print(f"🔍 Sampling checkpoints at epochs: {[epoch for epoch, _ in sample_epochs]}")
    
    # 分析特征演化趋势
    feature_evolution = analyze_feature_evolution(sample_epochs)
    
    # 可视化训练趋势
    visualize_training_trend(epoch_ckpts, feature_evolution)
    
    return epoch_ckpts

def analyze_feature_evolution(sample_epochs):
    """分析特征随训练epoch的演化"""
    print("🔄 Analyzing feature evolution across epochs...")
    
    evolution_data = {}
    
    # 使用固定的一组测试图像
    test_images = get_test_images(50)
    if not test_images:
        print("❌ No test images found for feature evolution analysis")
        return None
    
    for epoch, ckpt_file in tqdm(sample_epochs, desc="Analyzing epochs"):
        model_path = os.path.join(CKPT_DIR, ckpt_file)
        model = load_moco_encoder(model_path)
        
        features = []
        for img_path in test_images:
            try:
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor()
                ])
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0)
                with torch.no_grad():
                    feat = model.get_features(x).cpu().numpy().flatten()
                features.append(feat)
            except Exception:
                continue
        
        if features:
            features = np.array(features)
            # 计算特征统计量
            evolution_data[epoch] = {
                'mean_norm': np.mean(np.linalg.norm(features, axis=1)),
                'std_norm': np.std(np.linalg.norm(features, axis=1)),
                'feature_std': np.mean(np.std(features, axis=0)),  # 特征维度上的标准差
                'num_features': len(features)
            }
    
    return evolution_data

def get_test_images(n=50):
    """获取用于分析的测试图像"""
    img_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_paths.append(os.path.join(root, f))
            if len(img_paths) >= n:
                break
        if len(img_paths) >= n:
            break
    return img_paths[:n]

def visualize_training_trend(epoch_ckpts, feature_evolution):
    """可视化训练趋势"""
    if not feature_evolution:
        print("❌ No feature evolution data to visualize")
        return
    
    epochs = list(feature_evolution.keys())
    mean_norms = [feature_evolution[ep]['mean_norm'] for ep in epochs]
    feature_stds = [feature_evolution[ep]['feature_std'] for ep in epochs]
    
    # 创建训练趋势分析图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Stage-1 MoCo Training Trend Analysis (from Checkpoints)', fontsize=16, fontweight='bold')
    
    # 1. 特征范数变化
    axes[0, 0].plot(epochs, mean_norms, 'o-', color='#2E5C8A', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Feature Norm')
    axes[0, 0].set_title('Feature Norm Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征多样性变化
    axes[0, 1].plot(epochs, feature_stds, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Feature Diversity (Std)')
    axes[0, 1].set_title('Feature Diversity Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Checkpoint分布
    all_epochs = [epoch for epoch, _ in epoch_ckpts]
    axes[1, 0].hist(all_epochs, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Number of Checkpoints')
    axes[1, 0].set_title('Checkpoint Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 训练进度推断
    total_epochs = max(all_epochs)
    progress_ratio = len(epoch_ckpts) / total_epochs if total_epochs > 0 else 0
    
    labels = ['Saved', 'Missing']
    sizes = [len(epoch_ckpts), max(0, total_epochs - len(epoch_ckpts))]
    colors = ['#96CEB4', '#FFEAA7']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title(f'Checkpoint Coverage (Total: {total_epochs} epochs)')
    
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "training_trend_from_checkpoints.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training trend analysis to {save_path}")
    
    # 打印训练趋势报告
    print_training_trend_report(epoch_ckpts, feature_evolution)

def print_training_trend_report(epoch_ckpts, feature_evolution):
    """打印训练趋势报告"""
    print("\n" + "="*60)
    print("📈 TRAINING TREND ANALYSIS REPORT (From Checkpoints)")
    print("="*60)
    
    total_epochs = max([epoch for epoch, _ in epoch_ckpts])
    saved_epochs = len(epoch_ckpts)
    
    print(f"📊 Total training epochs: {total_epochs}")
    print(f"📊 Saved checkpoints: {saved_epochs}")
    print(f"📊 Checkpoint coverage: {saved_epochs/total_epochs*100:.1f}%")
    
    if feature_evolution:
        epochs = list(feature_evolution.keys())
        first_epoch = min(epochs)
        last_epoch = max(epochs)
        
        first_norm = feature_evolution[first_epoch]['mean_norm']
        last_norm = feature_evolution[last_epoch]['mean_norm']
        norm_change = last_norm - first_norm
        
        first_std = feature_evolution[first_epoch]['feature_std']
        last_std = feature_evolution[last_epoch]['feature_std']
        std_change = last_std - first_std
        
        print(f"📊 Feature norm change: {first_norm:.4f} → {last_norm:.4f} ({norm_change:+.4f})")
        print(f"📊 Feature diversity change: {first_std:.4f} → {last_std:.4f} ({std_change:+.4f})")
        
        # 趋势判断
        if norm_change > 0.1 and std_change > 0:
            trend_status = "✅ STRONG LEARNING"
        elif norm_change > 0.05:
            trend_status = "✅ GOOD LEARNING"
        elif norm_change > 0:
            trend_status = "⚠️ MODERATE LEARNING"
        else:
            trend_status = "❌ POOR LEARNING"
            
        print(f"📊 Learning trend: {trend_status}")
    
    print("="*60)

# ==============================
# 加载图片并提取特征
# ==============================
def extract_features(model, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    features, labels = [], []
    subdirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"📂 Detected datasets: {subdirs}")

    for d in subdirs:
        d_path = os.path.join(DATA_DIR, d)
        imgs = [os.path.join(d_path, f) for f in os.listdir(d_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not imgs: 
            continue
        for img_path in tqdm(imgs[:SAMPLE_LIMIT], desc=f"Extracting {d}"):
            from PIL import Image
            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.get_features(x).cpu().numpy().flatten()
                features.append(feat)
                labels.append(d)
            except Exception:
                continue

    feats = np.array(features)
    labels = np.array(labels)
    print(f"✅ Extracted features: {feats.shape}")
    return feats, labels

# ==============================
# 特征可视化
# ==============================
def visualize_features(feats, labels, method="tsne"):
    print(f"🔍 Reducing dimensions using {method.upper()}...")
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000) if method == "tsne" else PCA(n_components=2)
    reduced = reducer.fit_transform(feats)

    plt.figure(figsize=(10, 8))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=lab, alpha=0.7, s=25)
    plt.legend()
    plt.title(f"Feature Distribution ({method.upper()})")
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"features_{method}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved feature visualization to {save_path}")

# ==============================
# 特征一致性分析
# ==============================
def feature_consistency(model):
    from PIL import Image
    from torchvision import transforms

    print("🧩 Evaluating feature consistency under augmentations...")
    img_paths = get_test_images(50)

    aug = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    sims = []
    for p in tqdm(img_paths, desc="Evaluating"):
        img = Image.open(p).convert("RGB")
        base = transforms.Resize((IMG_SIZE, IMG_SIZE))(img)
        base_t = transforms.ToTensor()(base).unsqueeze(0)
        aug_t = aug(img).unsqueeze(0)
        with torch.no_grad():
            f1 = model.get_features(base_t).flatten()
            f2 = model.get_features(aug_t).flatten()
            sim = torch.nn.functional.cosine_similarity(f1, f2, dim=0).item()
        sims.append(sim)

    plt.figure(figsize=(8, 5))
    plt.hist(sims, bins=20, color="#FF8C42", edgecolor="black", alpha=0.7)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Feature Consistency under Augmentation")
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "feature_consistency.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved feature consistency analysis to {save_path}")
    
    print(f"📊 Feature consistency - Mean: {np.mean(sims):.4f}, Std: {np.std(sims):.4f}")
    print(f"📊 Feature consistency - Min: {np.min(sims):.4f}, Max: {np.max(sims):.4f}")

# ==============================
# 主函数
# ==============================
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")])
    if not ckpts:
        print("❌ No checkpoint found.")
        return
    
    # 分析训练趋势
    analyze_training_from_checkpoints()
    
    # 加载最终模型进行其他分析
    best_ckpt = os.path.join(CKPT_DIR, ckpts[-1])
    model = load_moco_encoder(best_ckpt)

    # 执行其他分析
    feats, labels = extract_features(model)
    visualize_features(feats, labels, "pca")
    visualize_features(feats, labels, "tsne")
    feature_consistency(model)
    
    print("✅ Stage-1 Performance Analysis Completed!")
    print("📊 Check the analysis_plots directory for detailed visualizations")
if __name__ == "__main__":
    main()