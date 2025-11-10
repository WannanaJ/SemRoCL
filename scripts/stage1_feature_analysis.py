"""
Additional analyses for Stage-1 representations:
- t-SNE / UMAP visualization using cached features.
- Illumination invariance test on paired low/high-light samples.
"""

import argparse
import json
import random
import sys
import pathlib
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

PathLike = pathlib.Path

REPO_ROOT = PathLike(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from data_loader import OptimizedLowLightDataset
from model.encoder_moco import MoCoV3Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 feature visualization and illumination analysis.")
    parser.add_argument(
        "--features",
        type=str,
        default="outputs/moco_pretrain_stage1/eval/extracted_features.npz",
        help="Cached feature file produced by stage1_feature_eval.py",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="outputs/moco_pretrain_stage1/eval/class_names.json",
        help="JSON file listing class names (same order as labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/moco_pretrain_stage1/eval",
        help="Directory to place analysis outputs.",
    )
    parser.add_argument("--tsne-sample", type=int, default=2000, help="Maximum samples for t-SNE / UMAP.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=42)
    parser.add_argument(
        "--run-umap",
        action="store_true",
        help="Attempt UMAP visualization (requires umap-learn).",
    )
    parser.add_argument(
        "--illumination-root",
        type=str,
        default="data/stage2_paired",
        help="Root directory of paired low/high datasets for illumination test.",
    )
    parser.add_argument("--illumination-datasets", type=str, nargs="*", default=["LOL-v1"])
    parser.add_argument("--illumination-samples", type=int, default=512)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/moco_pretrain_stage1/checkpoints/moco_pretrain_epoch_120.pth",
        help="Stage-1 encoder checkpoint (for illumination test).",
    )
    parser.add_argument("--base-encoder", type=str, default="resnet50")
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-illumination",
        action="store_true",
        help="Skip illumination invariance test.",
    )
    return parser.parse_args()


def load_cached_features(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["features"], data["labels"]


def maybe_load_class_names(path: PathLike) -> Optional[Sequence[str]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sample_subset(features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int):
    n = len(features)
    if n <= max_samples:
        return features, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return features[idx], labels[idx]


def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Sequence[str]],
    path: PathLike,
    title: str,
):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=14, cmap="tab20")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    if class_names is not None and len(class_names) <= 20:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_tsne(features: np.ndarray, labels: np.ndarray, args: argparse.Namespace, class_names):
    sampled_feats, sampled_labels = sample_subset(features, labels, args.tsne_sample, args.tsne_seed)
    tsne = TSNE(  # type: ignore[call-arg]
        n_components=2,
        perplexity=min(args.tsne_perplexity, len(sampled_feats) / 3 - 1e-5),
        random_state=args.tsne_seed,
        init="pca",
    )
    embedding = tsne.fit_transform(sampled_feats)
    plot_embedding(
        embedding,
        sampled_labels,
        class_names,
        PathLike(args.output_dir) / "stage1_tsne.png",
        title=f"Stage-1 Feature t-SNE (N={len(sampled_feats)})",
    )
    return embedding


def run_umap(features: np.ndarray, labels: np.ndarray, args: argparse.Namespace, class_names):
    try:
        import umap  # type: ignore[import]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Warning] UMAP not available: {exc}")
        return None

    sampled_feats, sampled_labels = sample_subset(features, labels, args.tsne_sample, args.tsne_seed)
    reducer = umap.UMAP(n_components=2, random_state=args.tsne_seed)  # type: ignore[call-arg]
    embedding = reducer.fit_transform(sampled_feats)
    plot_embedding(
        embedding,
        sampled_labels,
        class_names,
        PathLike(args.output_dir) / "stage1_umap.png",
        title=f"Stage-1 Feature UMAP (N={len(sampled_feats)})",
    )
    return embedding


def load_stage2_dataset(root: PathLike, subset_names: Sequence[str], img_size: int = 224):
    datasets_list = []
    for name in subset_names:
        subset_path = root / name
        if not subset_path.exists():
            print(f"[Warning] Illumination subset not found: {subset_path}")
            continue
        ds = OptimizedLowLightDataset(
            root_dir=str(subset_path),
            split="train",
            paired=True,
            img_size=img_size,
            augment=False,
            normalize=True,
            cache_images=False,
            use_pil=True,
        )
        if len(ds) > 0:
            datasets_list.append(ds)
    if not datasets_list:
        raise RuntimeError("No valid datasets for illumination analysis.")
    if len(datasets_list) == 1:
        return datasets_list[0]
    from torch.utils.data import ConcatDataset

    return ConcatDataset(datasets_list)


def build_encoder(args: argparse.Namespace, device: torch.device) -> MoCoV3Encoder:
    model = MoCoV3Encoder(
        base_encoder=args.base_encoder,
        feat_dim=args.feat_dim,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
    )
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


def illumination_invariance(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_encoder(args, device)
    dataset = load_stage2_dataset(PathLike(args.illumination_root), args.illumination_datasets, img_size=224)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    distances = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)
            emb_low = F.normalize(model.encoder_q(low, return_intermediate=False)[0], dim=1)
            emb_high = F.normalize(model.encoder_q(high, return_intermediate=False)[0], dim=1)
            cos_dist = 1.0 - torch.sum(emb_low * emb_high, dim=1)
            distances.extend(cos_dist.cpu().tolist())
            count += low.size(0)
            if count >= args.illumination_samples:
                break

    distances = np.array(distances[: args.illumination_samples])
    stats = {
        "num_pairs": int(len(distances)),
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "std_distance": float(distances.std()),
        "p90_distance": float(np.percentile(distances, 90)),
        "p95_distance": float(np.percentile(distances, 95)),
        "proportion_below_0.1": float((distances < 0.1).mean()),
    }

    out_dir = PathLike(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "illumination_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=40, color="steelblue", alpha=0.8)
    plt.axvline(0.1, color="red", linestyle="--", label="0.1 threshold")
    plt.title("Illumination Invariance - Cosine Distance Distribution")
    plt.xlabel("1 - cosine_similarity(low, high)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "illumination_distance_hist.png", dpi=200)
    plt.close()

    print("[Illumination Analysis]")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features_path = PathLike(args.features)
    output_dir = PathLike(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = load_cached_features(features_path)
    class_names = maybe_load_class_names(PathLike(args.class_names))

    print("[t-SNE]")
    run_tsne(features, labels, args, class_names)
    if args.run_umap:
        print("[UMAP]")
        run_umap(features, labels, args, class_names)

    if not args.skip_illumination:
        illumination_invariance(args)
    else:
        print("[Info] Skipping illumination analysis as requested.")

    print(f"Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
