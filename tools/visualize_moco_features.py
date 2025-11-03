import argparse
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image


# ==========================
# 模型加载
# ==========================
def load_encoder(weight_path):
    print(f"🔍 Loading pretrained weights: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu")

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Identity()

    if "encoder_q" in ckpt:
        print("✅ Detected MoCo-style checkpoint (encoder_q)")
        model.load_state_dict(ckpt["encoder_q"], strict=False)
    elif "model_state_dict" in ckpt:
        print("✅ Detected model_state_dict format")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "state_dict" in ckpt:
        print("✅ Detected plain state_dict format")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        print("⚠️ No recognized keys found; loading entire checkpoint directly")
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


# ==========================
# 特征提取
# ==========================
def extract_features(model, root_dir, sample_limit=200):
    all_features, all_labels = [], []
    print(f"📁 Scanning root directory: {root_dir}")

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for sub in subdirs:
        path = os.path.join(root_dir, sub)
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            images.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext.split('*')[-1])])
        images = images[:sample_limit]
        if not images:
            continue

        feats = []
        for img_path in tqdm(images, desc=f"🧩 Processing dataset: {sub}"):
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            x = torchvision.transforms.ToTensor()(img).unsqueeze(0)
            with torch.no_grad():
                f = model(x).squeeze().numpy()
            feats.append(f)

        all_features.append(np.vstack(feats))
        all_labels.extend([sub] * len(feats))

    if len(all_features) == 0:
        print("⚠️ No features extracted.")
        return np.array([]), []

    all_features = np.vstack(all_features)
    print(f"✅ Total features: {all_features.shape}")
    return all_features, all_labels


# ==========================
# 可视化模块
# ==========================
def visualize_features(features, labels, output_path, use_tsne=False):
    if len(features) == 0:
        print("⚠️ No feature data to visualize.")
        return

    print(f"🔍 Reducing dimensions using {'t-SNE' if use_tsne else 'PCA'} ...")
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, verbose=1) if use_tsne else PCA(n_components=2)
    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, lbl in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lbl]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], color=colors[i], label=lbl, s=10)

    plt.title("Feature Visualization (t-SNE/PCA)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Visualization saved to {output_path}")


# ==========================
# Loss & PSNR 曲线绘制
# ==========================
def plot_training_curves(log_path=None, metrics_path=None):
    if log_path and os.path.exists(log_path):
        print(f"📉 Plotting training loss from {log_path}")
        losses, epochs = [], []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Average Loss" in line:
                    parts = line.strip().split()
                    epoch = int(parts[1])
                    loss = float(parts[-1])
                    epochs.append(epoch)
                    losses.append(loss)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, label="Average Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Trend (Stage-1)")
        plt.grid(True)
        plt.legend()
        plt.savefig("outputs/plots/loss_trend.png", dpi=300)
        print("✅ Saved: outputs/plots/loss_trend.png")

    if metrics_path and os.path.exists(metrics_path):
        print(f"📈 Plotting PSNR curve from {metrics_path}")
        df = pd.read_csv(metrics_path)
        if "epoch" in df.columns and "psnr" in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(df["epoch"], df["psnr"], label="PSNR", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("PSNR (dB)")
            plt.title("PSNR Trend (Stage-2)")
            plt.grid(True)
            plt.legend()
            plt.savefig("outputs/plots/psnr_trend.png", dpi=300)
            print("✅ Saved: outputs/plots/psnr_trend.png")


# ==========================
# 主函数
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to pretrained weights (.pth)")
    parser.add_argument("--data_root", default="data/stage1_unpaired", help="Path to data root directory")
    parser.add_argument("--output", default="outputs/plots/features_tsne.png", help="Output figure path")
    parser.add_argument("--log_path", default=None, help="Training log path for loss curve")
    parser.add_argument("--metrics_path", default=None, help="Metrics CSV path for PSNR")
    parser.add_argument("--sample_limit", type=int, default=300, help="Number of samples per dataset")
    parser.add_argument("--use_tsne", action="store_true", help="Use t-SNE for visualization")
    args = parser.parse_args()

    model = load_encoder(args.weights)
    feats, labels = extract_features(model, args.data_root, args.sample_limit)
    visualize_features(feats, labels, args.output, use_tsne=args.use_tsne)
    plot_training_curves(args.log_path, args.metrics_path)


if __name__ == "__main__":
    main()
