#!/usr/bin/env python3
"""
Utility to visualise Stage-2 training metrics.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _aggregate_epoch_metrics(rows: List[Dict[str, str]], key: str) -> List[Tuple[int, float]]:
    bucket: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        bucket[int(row['epoch'])].append(float(value))
    return sorted((epoch, sum(vals) / len(vals)) for epoch, vals in bucket.items() if vals)


def load_metrics(path: str):
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['step'] == 'val':
                val_rows.append(row)
            else:
                train_rows.append(row)
    return train_rows, val_rows


def plot_curves(metrics_path: str, output_dir: str):
    train_rows, val_rows = load_metrics(metrics_path)

    train_psnr = _aggregate_epoch_metrics(train_rows, 'psnr')
    train_ssim = _aggregate_epoch_metrics(train_rows, 'ssim')
    train_lpips = _aggregate_epoch_metrics(train_rows, 'lpips')
    train_sem_cos = _aggregate_epoch_metrics(train_rows, 'semantic_cos')
    train_sem_ssim = _aggregate_epoch_metrics(train_rows, 'semantic_ssim')
    train_grad_g = _aggregate_epoch_metrics(train_rows, 'grad_g')
    train_grad_d = _aggregate_epoch_metrics(train_rows, 'grad_d')
    train_speed = _aggregate_epoch_metrics(train_rows, 'speed_imgs_sec')

    weights_color = _aggregate_epoch_metrics(train_rows, 'w_color')
    weights_semantic = _aggregate_epoch_metrics(train_rows, 'w_semantic')
    weights_perc = _aggregate_epoch_metrics(train_rows, 'w_perceptual')
    weights_freq = _aggregate_epoch_metrics(train_rows, 'w_freq')
    weights_adv = _aggregate_epoch_metrics(train_rows, 'w_adv')

    val_psnr = [(int(row['epoch']), float(row['psnr'])) for row in val_rows if row.get('psnr')]
    val_ssim = [(int(row['epoch']), float(row['ssim'])) for row in val_rows if row.get('ssim')]
    val_lpips = [(int(row['epoch']), float(row['lpips'])) for row in val_rows if row.get('lpips')]
    val_sem_cos = [(int(row['epoch']), float(row['semantic_cos'])) for row in val_rows if row.get('semantic_cos')]
    val_sem_ssim = [(int(row['epoch']), float(row['semantic_ssim'])) for row in val_rows if row.get('semantic_ssim')]

    os.makedirs(output_dir, exist_ok=True)

    def _plot(epoch_metric, label, ax, style='-'):
        if not epoch_metric:
            return
        epochs, values = zip(*epoch_metric)
        ax.plot(epochs, values, style, label=label)

    # Figure 1: PSNR/SSIM/LPIPS
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    _plot(train_psnr, 'Train PSNR', ax[0])
    _plot(val_psnr, 'Val PSNR', ax[0], '--')
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].legend()

    _plot(train_ssim, 'Train SSIM', ax[1])
    _plot(val_ssim, 'Val SSIM', ax[1], '--')
    ax[1].set_ylabel('SSIM')
    ax[1].legend()

    _plot(train_lpips, 'Train LPIPS', ax[2])
    _plot(val_lpips, 'Val LPIPS', ax[2], '--')
    ax[2].set_ylabel('LPIPS')
    ax[2].set_xlabel('Epoch')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'quality_curves.png'), dpi=200)
    plt.close(fig)

    # Figure 2: Curriculum weights
    fig, ax = plt.subplots(figsize=(8, 5))
    for data, label in [
        (weights_color, 'w_color'),
        (weights_semantic, 'w_semantic'),
        (weights_perc, 'w_perceptual'),
        (weights_freq, 'w_freq'),
        (weights_adv, 'w_adv'),
    ]:
        _plot(data, label, ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight value')
    ax.legend()
    ax.set_title('Curriculum Weights')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'curriculum_weights.png'), dpi=200)
    plt.close(fig)

    # Figure 3: Semantic consistency
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    _plot(train_sem_cos, 'Train Semantic Cosine', ax[0])
    _plot(val_sem_cos, 'Val Semantic Cosine', ax[0], '--')
    ax[0].set_ylabel('Cosine similarity')
    ax[0].legend()

    _plot(train_sem_ssim, 'Train Semantic SSIM', ax[1])
    _plot(val_sem_ssim, 'Val Semantic SSIM', ax[1], '--')
    ax[1].set_ylabel('Semantic SSIM')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'semantic_consistency.png'), dpi=200)
    plt.close(fig)

    # Figure 4: Gradients & speed
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    _plot(train_grad_g, 'Grad Norm G', ax[0])
    _plot(train_grad_d, 'Grad Norm D', ax[0], '--')
    ax[0].set_ylabel('Grad Norm')
    ax[0].legend()

    _plot(train_speed, 'Images/sec', ax[1])
    ax[1].set_ylabel('imgs/sec')

    iter_time = _aggregate_epoch_metrics(train_rows, 'iter_time_ms')
    _plot(iter_time, 'Iter time (ms)', ax[2])
    ax[2].set_ylabel('ms')
    ax[2].set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gradients_speed.png'), dpi=200)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV.")
    parser.add_argument('--metrics', type=str, required=True, help='Path to logs/metrics.csv')
    parser.add_argument('--output', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()
    plot_curves(args.metrics, args.output)


if __name__ == '__main__':
    main()
