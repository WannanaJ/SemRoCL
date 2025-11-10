"""
Stage-1 representation quality checks:
- Extract frozen encoder features on a labeled dataset (default: ExDark).
- Run linear probe (logistic regression) and k-NN evaluations.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from model.encoder_moco import MoCoV3Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 feature evaluation (linear probe + k-NN).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/moco_pretrain_stage1/checkpoints/moco_pretrain_epoch_120.pth",
        help="Path to the Stage-1 MoCo checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/stage3_eval/ExDark/ExDark",
        help="Path to labeled dataset root (ImageFolder structure).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/moco_pretrain_stage1/eval",
        help="Directory to store extracted features and reports.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-encoder", type=str, default="resnet50")
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument(
        "--knn-k", type=int, default=5, help="Number of neighbors for k-NN classification."
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing cached features if available.",
    )
    return parser.parse_args()


def build_dataloader(dataset_path: Path, batch_size: int, num_workers: int) -> Tuple[torch.utils.data.DataLoader, list]:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to [0, 1]
        ]
    )
    dataset = datasets.ImageFolder(str(dataset_path), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.classes


def load_encoder(args: argparse.Namespace, device: torch.device) -> MoCoV3Encoder:
    model = MoCoV3Encoder(
        base_encoder=args.base_encoder,
        feat_dim=args.feat_dim,
        queue_size=65536,
        momentum=0.999,
        temperature=0.07,
    )
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[Warning] Missing keys: {missing}")
        print(f"[Warning] Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


def extract_features(
    model: MoCoV3Encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            emb, _ = model.encoder_q(images, return_intermediate=False)
            emb = F.normalize(emb, dim=1)
            features.append(emb.cpu().numpy())
            labels.append(target.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def run_linear_probe(
    X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray
) -> Tuple[float, Dict[str, Any], Pipeline]:
    clf: Pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=None,
        ),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    report = cast(
        Dict[str, Any],
        classification_report(
            y_val,
            y_pred,
            output_dict=True,
            zero_division=0,  # type: ignore[arg-type]
        ),
    )
    return acc, report, clf


def run_knn(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    k: int,
) -> Dict[str, float]:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    top1_acc = float(knn.score(X_val, y_val))

    retriever = NearestNeighbors(
        n_neighbors=min(5, len(X_train)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(X_train)

    distances, indices = retriever.kneighbors(X_val)
    hits_top5 = 0
    for idx, true_label in zip(indices, y_val):
        neighbor_labels = y_train[idx]
        if true_label in neighbor_labels:
            hits_top5 += 1
    top5_acc = float(hits_top5 / len(y_val))

    return {"top1_accuracy": float(top1_acc), "top5_retrieval": float(top5_acc)}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cache = output_dir / "extracted_features.npz"
    if args.reuse_cache and feature_cache.exists():
        cache = np.load(feature_cache)
        features = np.asarray(cache["features"])
        labels = np.asarray(cache["labels"])
        class_names = json.loads((output_dir / "class_names.json").read_text())
        print(f"[Info] Loaded cached features from {feature_cache}")
    else:
        loader, class_names = build_dataloader(dataset_path, args.batch_size, args.num_workers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_encoder(args, device)
        features, labels = extract_features(model, loader, device)
        np.savez_compressed(feature_cache, features=features, labels=labels)
        (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
        print(f"[Info] Saved features to {feature_cache}")

    X_train, X_val, y_train, y_val = cast(
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        train_test_split(
            features,
            labels,
            test_size=args.test_ratio,
            random_state=args.seed,
            stratify=labels,
        ),
    )

    linear_probe_acc, report, linear_probe_model = run_linear_probe(X_train, X_val, y_train, y_val)
    knn_results = run_knn(X_train, X_val, y_train, y_val, args.knn_k)

    cm = confusion_matrix(y_val, linear_probe_model.predict(X_val))

    results = {
        "dataset": str(dataset_path),
        "num_samples": int(features.shape[0]),
        "num_classes": len(np.unique(labels)),
        "test_ratio": args.test_ratio,
        "linear_probe": {
            "top1_accuracy": linear_probe_acc,
            "classification_report": report,
        },
        "knn": knn_results,
    }

    report_path = output_dir / "linear_probe_knn_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(output_dir / "linear_probe_confusion_matrix.npy", cm)

    print("[Evaluation Results]")
    print(f"Samples: {results['num_samples']}, Classes: {results['num_classes']}")
    print(f"Linear Probe Top-1 Accuracy: {linear_probe_acc:.4f}")
    print(f"k-NN (k={args.knn_k}) Top-1 Accuracy: {knn_results['top1_accuracy']:.4f}")
    print(f"k-NN Top-5 Retrieval: {knn_results['top5_retrieval']:.4f}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
