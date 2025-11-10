"""
LaTeX Results Summary and LaTeX Table Generation Script
==================================================================================

TIPLaTeX
Generate standard LaTeX tables for TIP journal submission

LaTeX Generated LaTeX Tables:
- Main results table
- Ablation study table
- Detailed metrics comparison table

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


# ============================================
# Logging Configuration
# ============================================

def setup_logger(output_dir: str) -> logging.Logger:
    """ Setup logger"""
    log_file = os.path.join(output_dir, f'summarize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger('summarize')
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
    return df


# ============================================
# LaTeX LaTeX Table Generator
# ============================================

class LaTeXTableGenerator:
    """LaTeX LaTeX table generator"""

    def __init__(self, logger: logging.Logger):
        """
         Initialize

        Args:
            logger:  Logger
        """
        self.logger = logger

    @staticmethod
    def format_value(value: float, std: float = None, decimal: int = 2) -> str:
        """
        LaTeX Format value for LaTeX

        Args:
            value:  Value
            std: Standard deviation (optional)
            decimal:  Decimal places

        Returns:
             Formatted string
        """
        if pd.isna(value):
            return "---"

        if std is not None and not pd.isna(std):
            return f"{value:.{decimal}f} $\\pm$ {std:.{decimal}f}"
        else:
            return f"{value:.{decimal}f}"

    @staticmethod
    def bold_best(values: List[str], indices: List[int], higher_better: bool = True) -> List[str]:
        """
         Bold the best value

        Args:
            values:  List of values
            indices:  Indices to compare
            higher_better:  Whether higher is better

        Returns:
             List of bolded values
        """
        # Extract numeric part for comparison
        numeric_values = []
        for idx in indices:
            val_str = values[idx]
            if val_str == "---":
                numeric_values.append(float('-inf') if higher_better else float('inf'))
            else:
                # Extract first number
                numeric_part = val_str.split('$')[0].strip()
                try:
                    numeric_values.append(float(numeric_part))
                except:
                    numeric_values.append(float('-inf') if higher_better else float('inf'))

        # Find best value index
        if higher_better:
            best_local_idx = np.argmax(numeric_values)
        else:
            best_local_idx = np.argmin(numeric_values)

        best_global_idx = indices[best_local_idx]

        # Bold best value
        result = values.copy()
        result[best_global_idx] = f"\\textbf{{{values[best_global_idx]}}}"

        return result

    def generate_main_results_table(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
         Generate main results table

        Args:
            df:  DataFrame
            output_path:  Output path
        """
        self.logger.info("Generating main results table...")

        # HPI Sort by HPI
        df_sorted = df.sort_values('hpi', ascending=False).reset_index(drop=True)

        # LaTeX Generate LaTeX table
        latex_lines = []

        # Table begin
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Ablation Study Results on LOL-v1 Dataset. Best results are highlighted in \\textbf{bold}.}")
        latex_lines.append("\\label{tab:ablation_results}")
        latex_lines.append("\\begin{tabular}{l|cccccc|c}")
        latex_lines.append("\\hline")
        latex_lines.append("\\textbf{Method} & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ & \\textbf{NIQE}$\\downarrow$ & \\textbf{BRISQUE}$\\downarrow$ & \\textbf{$\\Delta$E}$\\downarrow$ & \\textbf{HPI}$\\uparrow$ \\\\")
        latex_lines.append("\\hline")

        # Prepare all rows data
        all_rows = []
        for idx, row in df_sorted.iterrows():
            method_name = row['ablation_name'].replace('_', '\\_')

            psnr = self.format_value(row.get('psnr', np.nan), row.get('psnr_std', np.nan), decimal=2)
            ssim = self.format_value(row.get('ssim', np.nan), row.get('ssim_std', np.nan), decimal=4)
            lpips = self.format_value(row.get('lpips', np.nan), row.get('lpips_std', np.nan), decimal=4)
            niqe = self.format_value(row.get('niqe', np.nan), row.get('niqe_std', np.nan), decimal=4)
            brisque = self.format_value(row.get('brisque', np.nan), row.get('brisque_std', np.nan), decimal=2)
            delta_e = self.format_value(row.get('delta_e', np.nan), row.get('delta_e_std', np.nan), decimal=2)
            hpi = self.format_value(row.get('hpi', np.nan), row.get('hpi_std', np.nan), decimal=4)

            all_rows.append([method_name, psnr, ssim, lpips, niqe, brisque, delta_e, hpi])

        # Bold best values
        num_rows = len(all_rows)
        indices = list(range(num_rows))

        # Bold each column
        for col_idx, higher_better in enumerate([True, True, False, False, False, False, True], start=1):
            column_values = [row[col_idx] for row in all_rows]
            bolded_values = self.bold_best(column_values, indices, higher_better)
            for row_idx, bolded_val in enumerate(bolded_values):
                all_rows[row_idx][col_idx] = bolded_val

        # Write table rows
        for row in all_rows:
            latex_lines.append(" & ".join(row) + " \\\\")

        # Table end
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        self.logger.info(f"Main results table saved to: {output_path}")

    def generate_ablation_components_table(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """
         Generate ablation components table

         Show contribution of each component

        Args:
            df:  DataFrame
            output_path:  Output path
        """
        self.logger.info("Generating ablation components table...")

        # Check if component info available
        if 'ablation_name' not in df.columns:
            self.logger.warning("No ablation_name column, skipping component table")
            return

        # Parse components
        # ablation_name "sem1_freq1_gan0_cur1"
        # Assume ablation_name format is "sem1_freq1_gan0_cur1"

        df_copy = df.copy()

        # Extract components
        df_copy['Semantic'] = df_copy['ablation_name'].str.contains('sem1').map({True: '\\checkmark', False: ''})
        df_copy['Frequency'] = df_copy['ablation_name'].str.contains('freq1').map({True: '\\checkmark', False: ''})
        df_copy['GAN'] = df_copy['ablation_name'].str.contains('gan1').map({True: '\\checkmark', False: ''})
        df_copy['Curriculum'] = df_copy['ablation_name'].str.contains('cur1').map({True: '\\checkmark', False: ''})

        # HPI Sort by HPI
        df_sorted = df_copy.sort_values('hpi', ascending=False).reset_index(drop=True)

        # LaTeX Generate LaTeX table
        latex_lines = []

        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Ablation Study: Effect of Different Components. \\checkmark indicates the component is enabled.}")
        latex_lines.append("\\label{tab:ablation_components}")
        latex_lines.append("\\begin{tabular}{cccc|ccc}")
        latex_lines.append("\\hline")
        latex_lines.append("\\textbf{Semantic} & \\textbf{Frequency} & \\textbf{GAN} & \\textbf{Curriculum} & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{HPI}$\\uparrow$ \\\\")
        latex_lines.append("\\hline")

        # Prepare all rows data
        all_rows = []
        for idx, row in df_sorted.iterrows():
            semantic = row['Semantic']
            frequency = row['Frequency']
            gan = row['GAN']
            curriculum = row['Curriculum']

            psnr = self.format_value(row.get('psnr', np.nan), decimal=2)
            ssim = self.format_value(row.get('ssim', np.nan), decimal=4)
            hpi = self.format_value(row.get('hpi', np.nan), decimal=4)

            all_rows.append([semantic, frequency, gan, curriculum, psnr, ssim, hpi])

        # Bold best values ()
        # Only bold numeric columns
        num_rows = len(all_rows)
        indices = list(range(num_rows))

        for col_idx, higher_better in zip([4, 5, 6], [True, True, True]):
            column_values = [row[col_idx] for row in all_rows]
            bolded_values = self.bold_best(column_values, indices, higher_better)
            for row_idx, bolded_val in enumerate(bolded_values):
                all_rows[row_idx][col_idx] = bolded_val

        # Write table rows
        for row in all_rows:
            latex_lines.append(" & ".join(row) + " \\\\")

        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        self.logger.info(f"Ablation components table saved to: {output_path}")

    def generate_comparison_table(
        self,
        df: pd.DataFrame,
        output_path: str,
        top_n: int = 5
    ) -> None:
        """
        N Generate detailed comparison table for top N configurations

        Args:
            df:  DataFrame
            output_path:  Output path
            top_n: N Top N
        """
        self.logger.info(f"Generating top {top_n} comparison table...")

        # N Select top N
        if 'hpi' not in df.columns:
            self.logger.warning("HPI column not found, cannot generate comparison table")
            return

        df_top = df.nlargest(top_n, 'hpi').reset_index(drop=True)

        # LaTeX Generate LaTeX table
        latex_lines = []

        latex_lines.append("\\begin{table*}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{Top {top_n} Configurations: Comprehensive Metrics Comparison.}}")
        latex_lines.append("\\label{tab:top_comparison}")
        latex_lines.append("\\begin{tabular}{l|cc|cc|cc|c}")
        latex_lines.append("\\hline")
        latex_lines.append("\\multirow{2}{*}{\\textbf{Configuration}} & \\multicolumn{2}{c|}{\\textbf{Full-Reference}} & \\multicolumn{2}{c|}{\\textbf{Perceptual}} & \\multicolumn{2}{c|}{\\textbf{No-Reference}} & \\multirow{2}{*}{\\textbf{HPI}$\\uparrow$} \\\\")
        latex_lines.append("\\cline{2-7}")
        latex_lines.append(" & \\textbf{PSNR}$\\uparrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{LPIPS}$\\downarrow$ & \\textbf{$\\Delta$E}$\\downarrow$ & \\textbf{NIQE}$\\downarrow$ & \\textbf{BRISQUE}$\\downarrow$ & \\\\")
        latex_lines.append("\\hline")

        # Prepare data
        all_rows = []
        for idx, row in df_top.iterrows():
            config_name = row['ablation_name'].replace('_', '\\_')

            psnr = self.format_value(row.get('psnr', np.nan), decimal=2)
            ssim = self.format_value(row.get('ssim', np.nan), decimal=4)
            lpips = self.format_value(row.get('lpips', np.nan), decimal=4)
            delta_e = self.format_value(row.get('delta_e', np.nan), decimal=2)
            niqe = self.format_value(row.get('niqe', np.nan), decimal=4)
            brisque = self.format_value(row.get('brisque', np.nan), decimal=2)
            hpi = self.format_value(row.get('hpi', np.nan), decimal=4)

            all_rows.append([config_name, psnr, ssim, lpips, delta_e, niqe, brisque, hpi])

        # Write table rows
        for row in all_rows:
            latex_lines.append(" & ".join(row) + " \\\\")

        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table*}")

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        self.logger.info(f"Top {top_n} comparison table saved to: {output_path}")


# ============================================
# Statistical Analysis
# ============================================

def generate_statistics_summary(
    df: pd.DataFrame,
    output_path: str,
    logger: logging.Logger
) -> None:
    """
     Generate statistics summary

    Args:
        df:  DataFrame
        output_path:  Output path
        logger:  Logger
    """
    logger.info("Generating statistics summary...")

    metrics = ['psnr', 'ssim', 'lpips', 'niqe', 'brisque', 'delta_e', 'hpi']
    available_metrics = [m for m in metrics if m in df.columns]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Statistical Summary of Ablation Study Results\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Number of configurations: {len(df)}\n")
        f.write(f"Dataset: LOL-v1\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=" * 100 + "\n")
        f.write("Metrics Statistics\n")
        f.write("=" * 100 + "\n\n")

        for metric in available_metrics:
            metric_upper = metric.upper()
            values = df[metric].dropna()

            if len(values) == 0:
                continue

            f.write(f"{metric_upper}:\n")
            f.write(f"  Mean:    {values.mean():.4f}\n")
            f.write(f"  Std:     {values.std():.4f}\n")
            f.write(f"  Min:     {values.min():.4f}\n")
            f.write(f"  Max:     {values.max():.4f}\n")
            f.write(f"  Median:  {values.median():.4f}\n")
            f.write(f"  Range:   {values.max() - values.min():.4f}\n")
            f.write("\n")

        # Best configuration
        if 'hpi' in df.columns:
            best_idx = df['hpi'].idxmax()
            best_row = df.loc[best_idx]

            f.write("=" * 100 + "\n")
            f.write("Best Configuration (Highest HPI)\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Name: {best_row['ablation_name']}\n")
            for metric in available_metrics:
                if metric in best_row:
                    f.write(f"{metric.upper():12s}: {best_row[metric]:.4f}\n")

    logger.info(f"Statistics summary saved to: {output_path}")


# ============================================
# Main Summarization Pipeline
# ============================================

def summarize_all_results(config_path: str) -> None:
    """
     Summarize all experiment results

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
    logger.info("SemRoCL Results Summarization for TIP Journal Submission")
    logger.info("=" * 80)

    # Load metrics data
    metrics_csv = os.path.join(output_base, 'metrics_summary.csv')
    if not os.path.exists(metrics_csv):
        logger.error(f"Metrics summary not found: {metrics_csv}")
        logger.error("Please run eval_metrics.py first!")
        return

    df = load_metrics_data(metrics_csv, logger)

    # LaTeX Create LaTeX directory
    latex_dir = os.path.join(output_base, 'latex_tables')
    os.makedirs(latex_dir, exist_ok=True)
    logger.info(f"LaTeX tables will be saved to: {latex_dir}")

    # Create generator
    generator = LaTeXTableGenerator(logger)

    # Generate various tables
    try:
        generator.generate_main_results_table(
            df,
            os.path.join(latex_dir, 'main_results_table.tex')
        )
    except Exception as e:
        logger.error(f"Failed to generate main results table: {e}")

    try:
        generator.generate_ablation_components_table(
            df,
            os.path.join(latex_dir, 'ablation_components_table.tex')
        )
    except Exception as e:
        logger.error(f"Failed to generate components table: {e}")

    try:
        generator.generate_comparison_table(
            df,
            os.path.join(latex_dir, 'top5_comparison_table.tex'),
            top_n=5
        )
    except Exception as e:
        logger.error(f"Failed to generate comparison table: {e}")

    # Generate statistics summary
    try:
        generate_statistics_summary(
            df,
            os.path.join(output_base, 'statistics_summary.txt'),
            logger
        )
    except Exception as e:
        logger.error(f"Failed to generate statistics summary: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Results summarization completed!")
    logger.info(f"LaTeX tables saved to: {latex_dir}")
    logger.info("=" * 80)
    logger.info("\nYou can now include these tables in your TIP paper:")
    logger.info("  \\input{latex_tables/main_results_table.tex}")
    logger.info("  \\input{latex_tables/ablation_components_table.tex}")
    logger.info("  \\input{latex_tables/top5_comparison_table.tex}")


# ============================================
# Command Line Interface
# ============================================

def parse_args():
    """ Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Summarize results and generate LaTeX tables for SemRoCL ablation study',
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
    summarize_all_results(args.config)


if __name__ == '__main__':
    main()
