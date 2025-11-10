"""
 Experiment Results Visualization Script
==========================================================

TIP
Generate high-quality plots and charts for TIP journal submission

 Generated Charts:
- Metrics comparison bar charts
- Metrics line plots
- Radar charts
- Heatmaps
- Error bar plots

 Author: SemRoCL Team
 Date: 2025-11-10
 Version: 1.0.0
"""

import os
import sys
import csv
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Setup Chinese font
try:
    from matplotlib.font_manager import FontProperties
    # Try system Chinese fonts
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# seaborn Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# matplotlib Set matplotlib parameters
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11


# ============================================
# Logging Configuration
# ============================================

def setup_logger(output_dir: str) -> logging.Logger:
    """ Setup logger"""
    log_file = os.path.join(output_dir, f'visualize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('visualize')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================
# Data Loading
# ============================================

def load_metrics_data(csv_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
     Load metrics data

    Args:
        csv_path: CSV CSV file path
        logger:  Logger

    Returns:
        pandas DataFrame
    """
    logger.info(f"Loading metrics data from: {csv_path}")

    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} ablation results")
    logger.info(f"Columns: {list(df.columns)}")

    return df


# ============================================
# Bar Charts
# ============================================

def plot_metrics_bars(
    df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
     Plot metrics comparison bar charts

    Args:
        df:  DataFrame
        output_dir:  Output directory
        logger:  Logger
    """
    logger.info("Generating bar charts...")

    metrics = ['psnr', 'ssim', 'lpips', 'niqe', 'hpi']
    metric_labels = {
        'psnr': 'PSNR (dB) ',
        'ssim': 'SSIM ',
        'lpips': 'LPIPS ',
        'niqe': 'NIQE ',
        'hpi': 'HPI '
    }

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in data, skipping")
            continue

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot bars
        x = np.arange(len(df))
        bars = ax.bar(
            x,
            df[metric],
            color=sns.color_palette("husl", len(df)),
            edgecolor='black',
            linewidth=1.2,
            alpha=0.8
        )

        # std/ Add error bars if std available
        if f'{metric}_std' in df.columns:
            ax.errorbar(
                x,
                df[metric],
                yerr=df[f'{metric}_std'],
                fmt='none',
                ecolor='black',
                elinewidth=1.5,
                capsize=4,
                alpha=0.7
            )

        # Set labels
        ax.set_xlabel('Ablation Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_labels.get(metric, metric.upper())} Comparison Across Ablations',
                     fontsize=16, fontweight='bold', pad=20)

        # x Set x ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['ablation_name'], rotation=45, ha='right', fontsize=10)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Display values on top of bars
        for i, (bar, value) in enumerate(zip(bars, df[metric])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + (df[f'{metric}_std'].iloc[i] if f'{metric}_std' in df.columns else 0),
                f'{value:.2f}' if metric != 'ssim' else f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
             )

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, f'{metric}_bar_chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved bar chart: {output_path}")


# ============================================
# Line Plots
# ============================================

def plot_metrics_lines(
    df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
     Plot metrics line plots

    Args:
        df:  DataFrame
        output_dir:  Output directory
        logger:  Logger
    """
    logger.info("Generating line plots...")

    # Main metrics
    main_metrics = ['psnr', 'ssim', 'lpips', 'hpi']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metric_labels = {
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'lpips': 'LPIPS',
        'hpi': 'HPI'
    }

    for idx, metric in enumerate(main_metrics):
        if metric not in df.columns:
            continue

        ax = axes[idx]
        x = np.arange(len(df))

        # Plot line
        ax.plot(
            x,
            df[metric],
            marker='o',
            markersize=8,
            linewidth=2.5,
            color='#2E86AB',
            markerfacecolor='#A23B72',
            markeredgecolor='white',
            markeredgewidth=2,
            label=metric_labels[metric]
        )

        # std/ Add error band if std available
        if f'{metric}_std' in df.columns:
            ax.fill_between(
                x,
                df[metric] - df[f'{metric}_std'],
                df[metric] + df[f'{metric}_std'],
                alpha=0.2,
                color='#2E86AB'
            )

        # Mark maximum
        if metric in ['psnr', 'ssim', 'hpi']:  # Higher is better
            max_idx = df[metric].idxmax()
            max_value = df[metric].max()
            ax.scatter(max_idx, max_value, s=200, color='red', marker='*', zorder=5,
                      edgecolors='black', linewidths=2, label='Best')
        else:  # Lower is better
            min_idx = df[metric].idxmin()
            min_value = df[metric].min()
            ax.scatter(min_idx, min_value, s=200, color='green', marker='*', zorder=5,
                      edgecolors='black', linewidths=2, label='Best')

        # Set labels
        ax.set_xlabel('Ablation Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]} Across Ablations',
                     fontsize=14, fontweight='bold', pad=15)

        # x Set x axis
        ax.set_xticks(x)
        ax.set_xticklabels(df['ablation_name'], rotation=45, ha='right', fontsize=9)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'metrics_line_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved line plots: {output_path}")


# ============================================
# Radar Chart
# ============================================

def plot_radar_chart(
    df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger,
    top_n: int = 5
) -> None:
    """
    N Plot radar chart comparing top N configurations

    Args:
        df:  DataFrame
        output_dir:  Output directory
        logger:  Logger
        top_n: N Show top N configurations
    """
    logger.info("Generating radar chart...")

    # NHPI Select top N by HPI
    if 'hpi' not in df.columns:
        logger.warning("HPI not found, cannot generate radar chart")
        return

    df_top = df.nlargest(top_n, 'hpi')

    # Normalize metrics
    metrics = ['psnr', 'ssim', 'lpips', 'niqe', 'hpi']
    metric_labels = ['PSNR', 'SSIM', 'LPIPS', 'NIQE', 'HPI']

    # [0, 1] Normalize to [0, 1]
    normalized_data = []
    for _, row in df_top.iterrows():
        values = []
        # PSNR normalized to [0, 1] assuming max 40 dB
        values.append(min(row.get('psnr', 0) / 40.0, 1.0))
        # SSIM:  [0, 1]
        values.append(row.get('ssim', 0))
        # LPIPS: 
        values.append(1.0 - row.get('lpips', 0))
        # NIQE normalized, lower is better
        values.append(max(1.0 - row.get('niqe', 0) / 10.0, 0.0))
        # HPI:  [0, 1]
        values.append(row.get('hpi', 0))
        normalized_data.append(values)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each configuration
    colors = sns.color_palette("husl", top_n)
    for i, (values, name) in enumerate(zip(normalized_data, df_top['ablation_name'])):
        values += values[:1]  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_title(f'Top {top_n} Configurations - Normalized Metrics Comparison',
                 fontsize=16, fontweight='bold', pad=30)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), framealpha=0.9, fontsize=10)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f'radar_chart_top{top_n}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved radar chart: {output_path}")


# ============================================
# Heatmap
# ============================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
     Plot metrics correlation heatmap

    Args:
        df:  DataFrame
        output_dir:  Output directory
        logger:  Logger
    """
    logger.info("Generating correlation heatmap...")

    # Select metric columns
    metrics = ['psnr', 'ssim', 'lpips', 'niqe', 'brisque', 'delta_e', 'hpi']
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 2:
        logger.warning("Not enough metrics for correlation analysis")
        return

    # Compute correlation
    corr_matrix = df[available_metrics].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'shrink': 0.8},
        ax=ax,
        vmin=-1,
        vmax=1
    )

    ax.set_title('Metrics Correlation Heatmap',
                 fontsize=16, fontweight='bold', pad=20)

    # Set labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved correlation heatmap: {output_path}")


# ============================================
# Combined Plot
# ============================================

def plot_comprehensive_comparison(
    df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
     Plot comprehensive comparison

    Args:
        df:  DataFrame
        output_dir:  Output directory
        logger:  Logger
    """
    logger.info("Generating comprehensive comparison plot...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. PSNR vs SSIM PSNR vs SSIM scatter
    ax1 = fig.add_subplot(gs[0, 0])
    if 'psnr' in df.columns and 'ssim' in df.columns:
        scatter = ax1.scatter(df['psnr'], df['ssim'], s=150, c=df.index,
                             cmap='viridis', edgecolors='black', linewidths=1.5, alpha=0.7)
        ax1.set_xlabel('PSNR (dB)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('SSIM', fontsize=11, fontweight='bold')
        ax1.set_title('PSNR vs SSIM', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # 2. LPIPS vs HPI LPIPS vs HPI scatter
    ax2 = fig.add_subplot(gs[0, 1])
    if 'lpips' in df.columns and 'hpi' in df.columns:
        scatter = ax2.scatter(df['lpips'], df['hpi'], s=150, c=df.index,
                             cmap='viridis', edgecolors='black', linewidths=1.5, alpha=0.7)
        ax2.set_xlabel('LPIPS', fontsize=11, fontweight='bold')
        ax2.set_ylabel('HPI', fontsize=11, fontweight='bold')
        ax2.set_title('LPIPS vs HPI', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # 3. HPI HPI ranking
    ax3 = fig.add_subplot(gs[0, 2])
    if 'hpi' in df.columns:
        df_sorted = df.sort_values('hpi', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
        ax3.barh(range(len(df_sorted)), df_sorted['hpi'], color=colors, edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(df_sorted)))
        ax3.set_yticklabels(df_sorted['ablation_name'], fontsize=9)
        ax3.set_xlabel('HPI', fontsize=11, fontweight='bold')
        ax3.set_title('HPI Ranking', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

    # 4-6.  Metrics box plots
    metrics_to_plot = [('psnr', 'PSNR (dB)'), ('ssim', 'SSIM'), ('lpips', 'LPIPS')]
    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[1, idx])
        if metric in df.columns:
            bp = ax.boxplot([df[metric]], vert=True, patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_edgecolor('black')
            bp['boxes'][0].set_linewidth(1.5)
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(['All Ablations'])

    # 7-9.  No-reference metrics distribution
    noreference_metrics = [('niqe', 'NIQE'), ('brisque', 'BRISQUE'), ('delta_e', 'Delta E')]
    for idx, (metric, label) in enumerate(noreference_metrics):
        ax = fig.add_subplot(gs[2, idx])
        if metric in df.columns:
            ax.hist(df[metric], bins=10, color='skyblue', edgecolor='black',
                   linewidth=1.2, alpha=0.7)
            ax.axvline(df[metric].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {df[metric].mean():.2f}')
            ax.set_xlabel(label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    # Save
    output_path = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comprehensive comparison: {output_path}")


# ============================================
# Main Visualization Pipeline
# ============================================

def visualize_all_results(config_path: str) -> None:
    """
     Visualize all experiment results

    Args:
        config_path:  Configuration file path
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    output_base = config['output']['base_dir']

    # Setup logger
    logger = setup_logger(output_base)
    logger.info("=" * 80)
    logger.info("SemRoCL Results Visualization for TIP Journal Submission")
    logger.info("=" * 80)

    # Load metrics data
    metrics_csv = os.path.join(output_base, 'metrics_summary.csv')
    if not os.path.exists(metrics_csv):
        logger.error(f"Metrics summary not found: {metrics_csv}")
        logger.error("Please run eval_metrics.py first!")
        return

    df = load_metrics_data(metrics_csv, logger)

    # plots Create plots directory
    plots_dir = os.path.join(output_base, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Plots will be saved to: {plots_dir}")

    # Generate various plots
    try:
        plot_metrics_bars(df, plots_dir, logger)
    except Exception as e:
        logger.error(f"Failed to generate bar charts: {e}")

    try:
        plot_metrics_lines(df, plots_dir, logger)
    except Exception as e:
        logger.error(f"Failed to generate line plots: {e}")

    try:
        plot_radar_chart(df, plots_dir, logger, top_n=5)
    except Exception as e:
        logger.error(f"Failed to generate radar chart: {e}")

    try:
        plot_correlation_heatmap(df, plots_dir, logger)
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")

    try:
        plot_comprehensive_comparison(df, plots_dir, logger)
    except Exception as e:
        logger.error(f"Failed to generate comprehensive comparison: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Visualization completed!")
    logger.info(f"All plots saved to: {plots_dir}")
    logger.info("=" * 80)


# ============================================
# Command Line Interface
# ============================================

def parse_args():
    """ Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize results for SemRoCL ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file (YAML)'
    )

    return parser.parse_args()


def main():
    """ Main function"""
    args = parse_args()
    visualize_all_results(args.config)


if __name__ == '__main__':
    main()
