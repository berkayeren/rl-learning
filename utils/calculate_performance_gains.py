#!/usr/bin/env python3
"""
Script to calculate performance gains from SQLite databases created by bootstraped_results.py.
Compares methods against a baseline (typically Count Based) and shows efficiency improvements.
Similar to the performance analysis presented in ADA/DoWhamV2 papers.
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import tabulate, fall back to simple printing if not available
try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not installed. Install with: pip install tabulate")
    print("Falling back to simple table output.\n")


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


def calculate_performance_gain(baseline_value, method_value, metric_type='episode_length'):
    """
    Calculate performance gain percentage.

    For metrics where lower is better (episode_length):
        gain = ((baseline - method) / baseline) * 100
        Positive gain = improvement (method is better)

    For metrics where higher is better (reward, percentage_visited):
        gain = ((method - baseline) / baseline) * 100
        Positive gain = improvement (method is better)
    """
    if baseline_value == 0:
        return np.nan

    # Determine if lower or higher is better based on metric type
    lower_is_better = ['episode_length', 'episode_len', 'length', 'steps']

    if any(metric in metric_type.lower() for metric in lower_is_better):
        # For episode length, lower is better
        gain = ((baseline_value - method_value) / baseline_value) * 100
    else:
        # For reward, percentage visited, etc., higher is better
        gain = ((method_value - baseline_value) / baseline_value) * 100

    return gain


def analyze_performance_gains(db_paths, labels, metric_name, baseline_label=None,
                              max_iterations=None, output_format='table', output_file=None):
    """
    Analyze performance gains across multiple experiments.

    Args:
        db_paths: List of paths to SQLite database files
        labels: List of labels for each database
        metric_name: Name of the metric to analyze
        baseline_label: Label of the baseline method (default: first label)
        max_iterations: Maximum iteration to analyze (optional)
        output_format: 'table', 'csv', or 'markdown'
        output_file: Path to save output (optional)
    """

    # Read all databases
    all_data = {}
    for db_path, label in zip(db_paths, labels):
        print(f"Reading database: {db_path} ({label})")
        df = read_sqlite_database(db_path)

        if df is None or df.empty:
            print(f"Warning: No data found in {db_path}")
            continue

        # Filter by metric
        df_metric = df[df['metric'] == metric_name]

        if df_metric.empty:
            print(f"Warning: Metric '{metric_name}' not found in {db_path}")
            continue

        # Filter by max iterations if specified
        if max_iterations is not None:
            df_metric = df_metric[df_metric['iteration'] <= max_iterations]

        all_data[label] = df_metric

    if not all_data:
        print("Error: No valid data found in any database")
        return None

    # Determine baseline
    if baseline_label is None:
        baseline_label = labels[0]
        print(f"Using '{baseline_label}' as baseline")

    if baseline_label not in all_data:
        print(f"Error: Baseline '{baseline_label}' not found in data")
        return None

    baseline_data = all_data[baseline_label]

    # Get all unique iterations across all methods
    all_iterations = sorted(set().union(*[set(df['iteration'].values) for df in all_data.values()]))

    # Build results table
    results = []
    for iteration in all_iterations:
        row = {'iteration': iteration}

        # Get baseline value
        baseline_row = baseline_data[baseline_data['iteration'] == iteration]
        if baseline_row.empty:
            continue

        baseline_mean = baseline_row['mean'].values[0]
        row[f'{baseline_label}_mean'] = baseline_mean

        # Calculate gains for each method
        for label in labels:
            if label == baseline_label:
                row[f'{label}_gain_%'] = 0.0
                continue

            if label not in all_data:
                row[f'{label}_mean'] = np.nan
                row[f'{label}_gain_%'] = np.nan
                continue

            method_data = all_data[label]
            method_row = method_data[method_data['iteration'] == iteration]

            if method_row.empty:
                row[f'{label}_mean'] = np.nan
                row[f'{label}_gain_%'] = np.nan
                continue

            method_mean = method_row['mean'].values[0]
            row[f'{label}_mean'] = method_mean

            gain = calculate_performance_gain(baseline_mean, method_mean, metric_name)
            row[f'{label}_gain_%'] = gain

        results.append(row)

    if not results:
        print("Error: No matching iterations found across methods")
        return None

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate summary statistics
    print("\n" + "=" * 80)
    print(f"Performance Gain Analysis: {metric_name}")
    print("=" * 80)

    # Print summary statistics for each method
    for label in labels:
        if label == baseline_label:
            continue

        gain_col = f'{label}_gain_%'
        if gain_col in results_df.columns:
            gains = results_df[gain_col].dropna()
            if len(gains) > 0:
                print(f"\n{label} vs {baseline_label}:")
                print(f"  Mean Gain: {gains.mean():.2f}%")
                print(f"  Median Gain: {gains.median():.2f}%")
                print(f"  Min Gain: {gains.min():.2f}%")
                print(f"  Max Gain: {gains.max():.2f}%")
                print(f"  Std Dev: {gains.std():.2f}%")

    print("\n" + "=" * 80)

    # Format output
    if output_format == 'table':
        # Create a clean table for display
        display_cols = ['iteration']
        for label in labels:
            display_cols.append(f'{label}_mean')
            if label != baseline_label:
                display_cols.append(f'{label}_gain_%')

        display_df = results_df[display_cols].copy()

        # Format numbers
        for col in display_df.columns:
            if col == 'iteration':
                continue
            elif 'gain' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")

        print("\nDetailed Results:")
        if HAS_TABULATE:
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        else:
            # Simple fallback printing
            print(display_df.to_string(index=False))

    elif output_format == 'csv':
        output_str = results_df.to_csv(index=False)
        print("\n" + output_str)

        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

    elif output_format == 'markdown':
        output_str = results_df.to_markdown(index=False)
        print("\n" + output_str)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"# Performance Gain Analysis: {metric_name}\n\n")
                f.write(f"**Baseline:** {baseline_label}\n\n")
                f.write(output_str)
            print(f"\nResults saved to: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate performance gains from SQLite experiment databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare DoWham and ADA against Count Based baseline
  python calculate_performance_gains.py \\
    --db_paths count_based.db dowham.db ada.db \\
    --labels "Count Based" "DoWham" "ADA" \\
    --metric episode_length_mean \\
    --baseline "Count Based"
  
  # Analyze first 30 iterations and save to CSV
  python calculate_performance_gains.py \\
    --db_paths baseline.db method1.db method2.db \\
    --labels Baseline Method1 Method2 \\
    --metric episode_reward_mean \\
    --max_iterations 30 \\
    --output_format csv \\
    --output results.csv
        """
    )

    parser.add_argument("--db_paths", nargs='+', required=True,
                        help="Paths to SQLite database files")
    parser.add_argument("--labels", nargs='+', required=True,
                        help="Labels for each database (must match number of db_paths)")
    parser.add_argument("--metric", required=True,
                        help="Metric name to analyze (e.g., episode_length_mean, episode_reward_mean)")
    parser.add_argument("--baseline",
                        help="Label of baseline method (default: first label)")
    parser.add_argument("--max_iterations", type=int,
                        help="Analyze up to this iteration (inclusive)")
    parser.add_argument("--output_format", choices=['table', 'csv', 'markdown'], default='table',
                        help="Output format (default: table)")
    parser.add_argument("--output",
                        help="Output file path (for csv or markdown formats)")

    args = parser.parse_args()

    # Validate arguments
    if len(args.db_paths) != len(args.labels):
        parser.error("Number of db_paths must match number of labels")

    if len(args.db_paths) < 2:
        parser.error("At least 2 databases required for comparison")

    # Run analysis
    analyze_performance_gains(
        db_paths=args.db_paths,
        labels=args.labels,
        metric_name=args.metric,
        baseline_label=args.baseline,
        max_iterations=args.max_iterations,
        output_format=args.output_format,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
