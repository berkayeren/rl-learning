#!/usr/bin/env python3
"""
Script to read SQLite databases created by bootstraped_results.py and plot multiple experiments on the same plot.
"""

import argparse
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from utils.bootstraped_results import calculate_confidence_intervals_over_time, extract_evaluation_metrics


def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for the data."""
    if len(data) <= 1:
        return data, (data[0] if len(data) == 1 else 0, data[0] if len(data) == 1 else 0)

    n = len(data)
    bootstrap_means = np.zeros(num_bootstrap_samples)

    # Generate bootstrap samples
    for i in range(num_bootstrap_samples):
        # Sample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    # Calculate confidence interval
    alpha = 1.0 - confidence_level
    lower_percentile = alpha / 2.0 * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return bootstrap_means, (lower_bound, upper_bound)


def read_sqlite_database(db_path):
    """Read experiment results from SQLite database."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(db_path)

    # Read the experiment_results table
    try:
        df = pd.read_sql_query("SELECT * FROM experiment_results", conn)
    except pd.errors.DatabaseError as e:
        print(f"Error reading from database {db_path}: {e}")
        conn.close()
        return None

    conn.close()
    return df


def process_experiments_from_db(df, metric_name, bin_size=None, confidence_level=0.95, num_bootstrap_samples=1000):
    """Process experiments from database and calculate confidence intervals."""
    if df is None or df.empty:
        return {}

    # Filter by metric if specified
    if metric_name:
        df = df[df['metric'] == metric_name]

    if df.empty:
        print(f"No data found for metric: {metric_name}")
        return {}

    return df


def plot_multiple_experiments(experiments_data, metric_name, confidence_level=0.95,
                              output_path=None, title=None, figsize=(12, 8)):
    """Plot multiple experiments on the same figure."""
    if not experiments_data:
        print("No experiment data to plot")
        return

    # Set up the plot with a nice color palette
    plt.style.use('default')
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))

    fig, ax = plt.subplots(figsize=figsize)

    for i, (exp_name, df) in enumerate(experiments_data.items()):
        if df.empty:
            continue

        color = colors[i]

        iterations = df['iteration']
        means = df['mean']
        lower_bounds = df['lower_bound']
        upper_bounds = df['upper_bound']

        # Plot confidence interval as filled area
        ax.fill_between(iterations, lower_bounds, upper_bounds,
                        color=color, alpha=0.3, label=f'{exp_name} ({int(100 * confidence_level)}% CI)')

        # Plot mean line
        ax.plot(iterations, means, color=color, linewidth=2, label=f'{exp_name} (Mean)')

    # Customize the plot
    ax.set_xlabel('Training Iterations', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{metric_name} Comparison Across Experiments', fontsize=14, fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Improve layout
    plt.tight_layout()

    # Save if output path specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from SQLite databases")
    parser.add_argument("--db_paths", nargs='+', required=True,
                        help="Paths to SQLite database files")
    parser.add_argument("--metric", required=True,
                        help="Metric name to plot")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for intervals (default: 0.95)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of bootstrap samples (default: 1000)")
    parser.add_argument("--bin_size", type=int,
                        help="Bin size for iteration grouping (optional)")
    parser.add_argument("--output",
                        help="Output file path for the plot (optional)")
    parser.add_argument("--title",
                        help="Custom title for the plot (optional)")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 8],
                        help="Figure size as width height (default: 12 8)")
    parser.add_argument("--ci_method", choices=['bootstrap', 'std_error'], default='bootstrap',
                        help="Method to calculate confidence intervals")

    args = parser.parse_args()

    # Read data from all databases
    all_experiments = []

    for db_path in args.db_paths:
        print(f"Reading database: {db_path}")
        df = read_sqlite_database(db_path)
        all_experiments.append(df)

    if not all_experiments:
        print("No experiment data found in any database")
        return

    distinctive_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    colors = [distinctive_colors[i % len(distinctive_colors)] for i in range(len(all_experiments))]
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for i, df in enumerate(all_experiments):
        color = colors[i]
        # 3) Plot single mean curve + its bootstrap CI
        iterations = df['iteration']
        m = df['mean']
        # lb = df['lower_bound']
        # ub = df['upper_bound']
        #
        # ax.fill_between(iterations, lb, ub, color=color, alpha=0.5,
        #                 label=f'{int(100 * args.confidence)}% CI', zorder=1)
        # ax.plot(iterations, lb, linestyle='--', color=color, linewidth=2, zorder=2, label='Lower CI')
        # ax.plot(iterations, ub, linestyle='--', color=color, linewidth=2, zorder=2, label='Upper CI')
        ax.plot(iterations, m, color=color, linewidth=3, zorder=3, label='Mean')

    ax.set_xlabel('Training Iterations')
    ax.set_ylabel(args.metric)
    ax.set_title(f'{args.metric} Over Training Iterations (Mean ± {int(100 * args.confidence)}% CI)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
