"""

Cross-Dataset Results Summarization Script

TIP
Generate cross-dataset comparison tables and plots for TIP paper
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logger(output_dir: str) -> logging.Logger:
    """"""
    log_file = os.path.join(output_dir, f'cross_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('cross_dataset')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def collect_dataset_results(base_dir: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """
    

    Args:
        base_dir: 
        logger: 

    Returns:
        : {dataset_name: metrics_df}
    """
    results = {}

    # 
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)

        if not os.path.isdir(dataset_path):
            continue

        # metrics_summary.csv
        metrics_file = os.path.join(dataset_path, 'metrics_summary.csv')

        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                results[dataset_dir] = df
                logger.info(f"Loaded results for {dataset_dir}: {len(df)} ablations")
            except Exception as e:
                logger.error(f"Failed to load {metrics_file}: {e}")

    logger.info(f"Collected results from {len(results)} datasets")
    return results


def generate_cross_dataset_table(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    

    Args:
        results: 
        output_dir: 
        logger: 
    """
    logger.info("Generating cross-dataset comparison table...")

    # 
    summary_data = []

    for dataset_name, df in results.items():
        # (HPI)
        if 'hpi' in df.columns:
            best_idx = df['hpi'].idxmax()
            best_row = df.loc[best_idx]

            summary_data.append({
                'Dataset': dataset_name,
                'Best Config': best_row['ablation_name'],
                'PSNR': f"{best_row.get('psnr', 0):.2f}",
                'SSIM': f"{best_row.get('ssim', 0):.4f}",
                'LPIPS': f"{best_row.get('lpips', 0):.4f}",
                'NIQE': f"{best_row.get('niqe', 0):.4f}",
                'HPI': f"{best_row.get('hpi', 0):.4f}"
            })

    summary_df = pd.DataFrame(summary_data)

    # CSV
    csv_path = os.path.join(output_dir, 'cross_dataset_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved cross-dataset summary: {csv_path}")

    # LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Cross-Dataset Performance Comparison. Best configuration for each dataset.}")
    latex_lines.append("\\label{tab:cross_dataset}")
    latex_lines.append("\\begin{tabular}{l|l|ccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Dataset} & \\textbf{Best Config} & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ & \\textbf{NIQE}$\\downarrow$ & \\textbf{HPI}$\\uparrow$ \\\\")
    latex_lines.append("\\hline")

    for _, row in summary_df.iterrows():
        dataset = row['Dataset'].replace('_', '\\_')
        config = row['Best Config'].replace('_', '\\_')
        latex_lines.append(
            f"{dataset} & {config} & {row['PSNR']} & {row['SSIM']} & "
            f"{row['LPIPS']} & {row['NIQE']} & {row['HPI']} \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_path = os.path.join(output_dir, 'cross_dataset_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))

    logger.info(f"Saved LaTeX table: {latex_path}")


def plot_cross_dataset_comparison(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    

    Args:
        results: 
        output_dir: 
        logger: 
    """
    logger.info("Generating cross-dataset comparison plots...")

    metrics = ['psnr', 'ssim', 'lpips', 'hpi']
    metric_labels = {
        'psnr': 'PSNR (dB) ',
        'ssim': 'SSIM ',
        'lpips': 'LPIPS ',
        'hpi': 'HPI '
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # 
        datasets = []
        values = []

        for dataset_name, df in results.items():
            if metric in df.columns and 'hpi' in df.columns:
                best_idx = df['hpi'].idxmax()
                best_value = df.loc[best_idx, metric]

                datasets.append(dataset_name)
                values.append(best_value)

        # 
        if datasets:
            colors = sns.color_palette("husl", len(datasets))
            bars = ax.bar(range(len(datasets)), values, color=colors, edgecolor='black', linewidth=1.2)

            # 
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() 2.,
                    height,
                    f'{value:.2f}' if metric == 'psnr' else f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_labels[metric]} Across Datasets',
                        fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'cross_dataset_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison plot: {plot_path}")


def generate_detailed_report(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    

    Args:
        results: 
        output_dir: 
        logger: 
    """
    logger.info("Generating detailed report...")

    report_path = os.path.join(output_dir, 'cross_dataset_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SemRoCL Cross-Dataset Performance Report\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of datasets: {len(results)}\n\n")

        for dataset_name, df in results.items():
            f.write("=" * 100 + "\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write("=" * 100 + "\n\n")

            if 'hpi' in df.columns:
                # 
                best_idx = df['hpi'].idxmax()
                best_row = df.loc[best_idx]

                f.write(f"Best Configuration: {best_row['ablation_name']}\n\n")

                f.write("Metrics:\n")
                f.write(f"  PSNR:     {best_row.get('psnr', 0):.2f} dB\n")
                f.write(f"  SSIM:     {best_row.get('ssim', 0):.4f}\n")
                f.write(f"  LPIPS:    {best_row.get('lpips', 0):.4f}\n")
                f.write(f"  NIQE:     {best_row.get('niqe', 0):.4f}\n")
                f.write(f"  BRISQUE:  {best_row.get('brisque', 0):.2f}\n")
                f.write(f"  Delta E:  {best_row.get('delta_e', 0):.2f}\n")
                f.write(f"  HPI:      {best_row.get('hpi', 0):.4f}\n\n")

                # 
                f.write("Statistics across all configurations:\n")
                for metric in ['psnr', 'ssim', 'lpips', 'hpi']:
                    if metric in df.columns:
                        values = df[metric].dropna()
                        f.write(f"  {metric.upper():8s}: mean={values.mean():.4f}, "
                               f"std={values.std():.4f}, "
                               f"min={values.min():.4f}, "
                               f"max={values.max():.4f}\n")

                f.write("\n")

    logger.info(f"Saved detailed report: {report_path}")


def main():
    """"""
    parser = argparse.ArgumentParser(
        description='Summarize cross-dataset ablation study results'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        required=True,
        help='Base directory containing all dataset results'
    )

    args = parser.parse_args()

    # 
    output_dir = os.path.join(args.base_dir, 'cross_dataset_summary')
    os.makedirs(output_dir, exist_ok=True)

    # 
    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info("Cross-Dataset Results Summarization")
    logger.info("=" * 80)

    # 
    results = collect_dataset_results(args.base_dir, logger)

    if not results:
        logger.error("No results found!")
        return

    # 
    generate_cross_dataset_table(results, output_dir, logger)

    # 
    plot_cross_dataset_comparison(results, output_dir, logger)

    # 
    generate_detailed_report(results, output_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Cross-dataset summarization completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
