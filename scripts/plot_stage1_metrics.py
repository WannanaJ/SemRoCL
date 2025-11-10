"""Generate visualization plots for Stage 1 MoCo training metrics."""
from __future__ import annotations
import csv
import pathlib
from collections import defaultdict, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(data, window: int):
    if window <= 1:
        return data[:]
    acc = []
    running = 0.0
    q = deque()
    for value in data:
        q.append(value)
        running += value
        if len(q) > window:
            running -= q.popleft()
        acc.append(running / len(q))
    return acc


def _safe_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_metrics(csv_path: pathlib.Path):
    steps, loss, pos, neg, grad, speed = [], [], [], [], [], []
    iter_time, compute_time, data_time = [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            pos.append(float(row["pos_sim_mean"]))
            neg.append(float(row["neg_sim_mean"]))
            grad.append(_safe_float(row.get("grad_norm")))
            speed.append(float(row["speed_imgs_sec"]))
            iter_time.append(_safe_float(row.get("iter_time_ms")))
            compute_time.append(_safe_float(row.get("compute_time_ms")))
            data_time.append(_safe_float(row.get("data_time_ms")))
    return steps, loss, pos, neg, grad, speed, iter_time, compute_time, data_time


def build_epoch_summary(csv_path: pathlib.Path):
    stats = defaultdict(lambda: {"loss": [], "pos": [], "neg": [], "grad": []})
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            stats[epoch]["loss"].append(float(row["loss"]))
            stats[epoch]["pos"].append(float(row["pos_sim_mean"]))
            stats[epoch]["neg"].append(float(row["neg_sim_mean"]))
            stats[epoch]["grad"].append(float(row["grad_norm"]))
    epochs = sorted(stats.keys())
    loss = [sum(stats[e]["loss"]) / len(stats[e]["loss"]) for e in epochs]
    pos = [sum(stats[e]["pos"]) / len(stats[e]["pos"]) for e in epochs]
    neg = [sum(stats[e]["neg"]) / len(stats[e]["neg"]) for e in epochs]
    grad = [sum(stats[e]["grad"]) / len(stats[e]["grad"]) for e in epochs]
    return epochs, loss, pos, neg, grad


def main():
    csv_path = pathlib.Path("outputs/moco_pretrain_stage1/logs/metrics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    out_dir = pathlib.Path("outputs/moco_pretrain_stage1/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, loss, pos, neg, grad, speed, iter_time, compute_time, data_time = load_metrics(csv_path)
    smooth_window = 200
    loss_s = moving_average(loss, smooth_window)
    pos_s = moving_average(pos, smooth_window)
    neg_s = moving_average(neg, smooth_window)
    grad_s = moving_average(grad, smooth_window)
    speed_s = moving_average(speed, smooth_window)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()

    axes[0].plot(steps, loss, color="lightgray", linewidth=0.6, label="Raw")
    axes[0].plot(steps, loss_s, color="tab:red", label=f"MA{smooth_window}")
    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(steps, pos, color="lightgray", linewidth=0.6, label="Raw")
    axes[1].plot(steps, pos_s, color="tab:blue", label=f"MA{smooth_window}")
    axes[1].set_title("Positive Pair Similarity")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_ylim(0.6, 1.01)
    axes[1].legend()

    axes[2].plot(steps, neg, color="lightgray", linewidth=0.6, label="Raw")
    axes[2].plot(steps, neg_s, color="tab:orange", label=f"MA{smooth_window}")
    axes[2].set_title("Negative Pair Similarity")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].legend()

    axes[3].plot(steps, grad, color="lightgray", linewidth=0.6, label="Grad Raw")
    axes[3].plot(steps, grad_s, color="tab:green", label=f"Grad MA{smooth_window}")
    axes[3].set_title("Gradient Norm & Throughput")
    axes[3].set_ylabel("Grad Norm")
    ax_speed = axes[3].twinx()
    ax_speed.plot(
        steps,
        speed_s,
        color="tab:purple",
        linestyle="--",
        label=f"Speed MA{smooth_window}",
    )
    ax_speed.set_ylabel("Images/sec")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Global Step")

    axes[3].legend(loc="upper right")
    ax_speed.legend(loc="lower right")

    axes[4].plot(steps, iter_time, color="tab:red", label="Iter (ms)")
    axes[4].plot(steps, compute_time, color="tab:green", label="Compute (ms)", alpha=0.8)
    axes[4].plot(steps, data_time, color="tab:blue", label="Data (ms)", alpha=0.8)
    axes[4].set_title("Timing per Iteration")
    axes[4].set_ylabel("Milliseconds")
    axes[4].legend()

    axes[5].plot(steps, speed, color="tab:purple")
    axes[5].set_title("Instant Throughput")
    axes[5].set_ylabel("Images/sec")

    for idx in (4, 5):
        axes[idx].grid(True, linestyle="--", alpha=0.3)
        axes[idx].set_xlabel("Global Step")

    fig.tight_layout()
    fig.savefig(out_dir / "stage1_training_overview.png", dpi=200)
    plt.close(fig)

    epochs, loss_epoch, pos_epoch, neg_epoch, grad_epoch = build_epoch_summary(csv_path)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.plot(epochs, loss_epoch, label="Loss", color="tab:red")
    ax2.plot(epochs, pos_epoch, label="Pos Sim", color="tab:blue")
    ax2.plot(epochs, neg_epoch, label="Neg Sim", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "stage1_epoch_trends.png", dpi=200)
    plt.close(fig2)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
