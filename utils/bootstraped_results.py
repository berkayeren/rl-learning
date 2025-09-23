import argparse
import os
import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def load_experiment_data(exp_dir, metric, min_timesteps=50_000_000, seed=None):
    """Load evaluation data from experiment directories, stopping at min_timesteps if provided."""
    result_directory = os.path.expanduser('~') + "/ray_results"
    full_path = os.path.join(result_directory, exp_dir)
    result_json = os.path.join(full_path, "result.json")

    # Extract seed from params.json
    if seed is None:
        params_path = os.path.join(full_path, 'params.json')
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                seed = params.get('seed', None)
            except Exception as e:
                print(f"Error reading seed from {params_path}: {e}")
        else:
            print(f"params.json not found in {full_path}")

    if os.path.exists(result_json):
        data = []
        reached = False
        with open(result_json, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            if line.strip():
                try:
                    entry = json.loads(line)
                    # Only append if timesteps_total is not past min_timesteps
                    if 'timesteps_total' in entry and entry['timesteps_total'] >= min_timesteps:
                        reached = True
                        break
                    data.append(entry)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line.strip()}")
                    continue

        df = pd.DataFrame(data)
        # Add seed as a column to the dataframe
        df['seed'] = seed
        return df
    else:
        print("result.json not found in", full_path)
        return None


def extract_evaluation_metrics(experiments, metric_name, iteration_column="training_iteration"):
    """Extract evaluation metrics from experiments using nested JSON structure.
       It retrieves the metric from ["evaluation"]["env_runners"][f"{metric_name}_mean"].
    """
    metrics_data = {}

    for exp_name, df in experiments.items():
        # Check if the nested column exists
        if "evaluation" in df.columns:
            def extract_value(row):
                eval_obj = row["evaluation"]
                if isinstance(eval_obj, dict):
                    env_runners = eval_obj.get("env_runners", {})
                    # Try main metric
                    val = env_runners.get(f"{metric_name}_mean", np.nan)
                    if not np.isnan(val):
                        return val
                    # Try custom_metrics if not found
                    custom_metrics = env_runners.get("custom_metrics", {})
                    return custom_metrics.get(f"{metric_name}_mean", np.nan)
                return np.nan

            df["value"] = df.apply(extract_value, axis=1)
            # Include seed column if it exists
            columns_to_keep = [iteration_column, "value", "timesteps_total", "seed"]

            sub = df[columns_to_keep].dropna()
            sub = sub.rename(columns={iteration_column: "iteration"})
            metrics_data[exp_name] = sub
        else:
            print(f"Metric {metric_name} not found in {exp_name}")
            continue

    return metrics_data


def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for the data."""
    n = len(data)
    bootstrap_means = np.zeros(num_bootstrap_samples)

    # Generate bootstrap samples with tqdm progress bar
    for i in tqdm(range(num_bootstrap_samples), desc="Bootstrapping"):
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


def calculate_bootstrap_ci_for_metrics(metrics_data, num_bootstrap_samples=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics."""
    results = {}

    for exp_name, df in metrics_data.items():
        metric_values = df.iloc[:, 1].values  # Metric column is the second one
        means, ci = bootstrap_confidence_interval(
            metric_values,
            num_bootstrap_samples=num_bootstrap_samples,
            confidence_level=confidence_level
        )

        results[exp_name] = {
            'mean': np.mean(metric_values),
            'bootstrap_mean': np.mean(means),
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }

    return results


def calculate_confidence_intervals_over_time(metrics_data, bin_size=None, method='bootstrap',
                                             num_bootstrap_samples=1000, confidence_level=0.95):
    """
    Calculate confidence intervals for metrics at each iteration or binned iterations.

    Args:
        metrics_data: Dictionary of dataframes with iterations and metric values
        bin_size: If provided, bin iterations into groups of this size
        method: Method to calculate confidence intervals ('bootstrap' or 'std_error')
        num_bootstrap_samples: Number of bootstrap samples if using bootstrap method
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary of processed dataframes with iterations, means, and confidence intervals
    """
    results = {}

    for exp_name, df in metrics_data.items():
        # Force using our renamed columns
        iteration_col = "iteration"
        metric_col = "value"

        # If bin_size is provided, bin the data
        if bin_size:
            df['bin'] = (df[iteration_col] // bin_size) * bin_size
            grouped = df.groupby('bin')
            iterations = np.array(grouped.first().index)
            values = [group[metric_col].values for _, group in grouped]
        else:
            # Process each unique iteration
            grouped = df.groupby(iteration_col)
            iterations = np.array(grouped.groups.keys())
            values = [group[metric_col].values for _, group in grouped]

        means = np.array([np.mean(v) for v in values])
        lower_bounds = np.zeros_like(means)
        upper_bounds = np.zeros_like(means)

        # Calculate confidence intervals for each iteration
        for i, vals in enumerate(tqdm(values, desc=f"Bootstrapping CI for {exp_name}")):
            if len(vals) <= 1:
                # Not enough data for CI calculation
                lower_bounds[i] = means[i]
                upper_bounds[i] = means[i]
                continue

            if method == 'bootstrap':
                # Bootstrap method
                bootstrap_means = np.zeros(num_bootstrap_samples)
                n = len(vals)

                for j in tqdm(range(num_bootstrap_samples), desc=f"Bootstrap samples for iter={i}", leave=False):
                    sample = np.random.choice(vals, size=n, replace=True)
                    bootstrap_means[j] = np.mean(sample)

                alpha = 1.0 - confidence_level
                lower_percentile = alpha / 2.0 * 100
                upper_percentile = (1.0 - alpha / 2.0) * 100
                lower_bounds[i] = np.percentile(bootstrap_means, lower_percentile)
                upper_bounds[i] = np.percentile(bootstrap_means, upper_percentile)
            else:
                # Standard error method with t-distribution
                sem = stats.sem(vals)
                n = len(vals)
                t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
                lower_bounds[i] = means[i] - t_val * sem
                upper_bounds[i] = means[i] + t_val * sem

        # Create result dataframe
        result_df = pd.DataFrame({
            'iteration': iterations,
            'mean': means,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        })

        # Sort by iteration to ensure correct plotting order
        result_df = result_df.sort_values('iteration')
        results[exp_name] = result_df

    return results


def calculate_average_confidence_intervals_over_time(metrics_data, bin_size=None,
                                                     num_bootstrap_samples=1000,
                                                     confidence_level=0.95):
    """Compute overall mean ± bootstrap CI across runs at each iteration/bin."""
    # 1) Extract each run's mean‐series
    run_means = {}
    for exp_name, df in metrics_data.items():
        # df now has columns ['iteration','mean','lower_bound','upper_bound']
        s = df.set_index('iteration')['mean']
        run_means[exp_name] = s

    # 2) Align on union of all iterations, then fill missing per‐run means
    all_iterations = sorted(set().union(*(s.index for s in run_means.values())))
    runs_df = pd.DataFrame(
        {exp: run_means[exp].reindex(all_iterations) for exp in run_means},
        index=all_iterations
    ).sort_index().ffill().bfill()

    # 3) Bootstrap across runs at each iteration
    means, lower_bounds, upper_bounds = [], [], []
    for vals in runs_df.values:
        m = vals.mean()
        bs, (lb, ub) = bootstrap_confidence_interval(
            vals,
            num_bootstrap_samples=num_bootstrap_samples,
            confidence_level=confidence_level
        )
        means.append(m)
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    return pd.DataFrame({
        'iteration': all_iterations,
        'mean': means,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate bootstrap confidence intervals for Ray experiments")
    parser.add_argument("--exp_dirs", nargs='+', required=True, help="List of experiment directories")
    parser.add_argument("--metric", required=True, help="Metric name to analyze")
    parser.add_argument("--samples", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--bin_size", type=int, help="Bin size for iteration grouping (optional)")
    parser.add_argument("--ci_method", choices=['bootstrap', 'std_error'], default='bootstrap',
                        help="Method to calculate confidence intervals")
    parser.add_argument("--sqlite_db", help="Path to SQLite database file to store results (optional)")
    parser.add_argument("--min_timesteps", type=int, default=50_000_000,
                        help="Minimum timesteps to load from result.json (default: 50M)")
    args = parser.parse_args()

    # Load data from each experiment
    experiments = {}
    seed = 0
    for exp_dir in args.exp_dirs:
        print(f"Loading data from {exp_dir[:150]}...")
        df = load_experiment_data(exp_dir, args.metric, min_timesteps=args.min_timesteps, seed=seed)
        seed += 1  # Increment seed for each experiment
        if df is not None:
            # Use strip to remove trailing slashes, then get basename
            exp_name = os.path.basename(exp_dir.rstrip('/'))
            experiments[exp_name] = df
        del df

    # Extract evaluation metrics
    metrics_data = extract_evaluation_metrics(experiments, args.metric)

    # 1) Compute per‐experiment CI over time (fills mean, lower_bound, upper_bound)
    metrics_data = calculate_confidence_intervals_over_time(
        metrics_data,
        bin_size=args.bin_size,
        method=args.ci_method,
        num_bootstrap_samples=args.samples,
        confidence_level=args.confidence
    )

    # 2) Compute overall average ± CI across those run‐means
    avg_df = calculate_average_confidence_intervals_over_time(
        metrics_data,
        bin_size=None,  # already binned above if needed
        num_bootstrap_samples=args.samples,
        confidence_level=args.confidence
    )

    # Optionally write results to SQLite
    if args.sqlite_db:
        conn = sqlite3.connect(args.sqlite_db)
        avg_df.to_sql('experiment_results', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Results written to SQLite database: {args.sqlite_db}")

    # 3) Plot single mean curve + its bootstrap CI
    fig, ax = plt.subplots(figsize=(12, 8))
    iterations = avg_df['iteration']
    m = avg_df['mean']
    lb = avg_df['lower_bound']
    ub = avg_df['upper_bound']

    # Use a smooth blue color palette
    color = '#1f77b4'  # Matplotlib default blue
    ci_fill = '#aec7e8'  # Lighter blue for CI fill
    ci_line = '#4a90e2'  # Medium blue for CI bounds

    ax.fill_between(iterations, lb, ub, color=ci_fill, alpha=0.5,
                    label=f'{int(100 * args.confidence)}% CI', zorder=1)
    ax.plot(iterations, lb, linestyle='--', color=ci_line, linewidth=2, zorder=2, label='Lower CI')
    ax.plot(iterations, ub, linestyle='--', color=ci_line, linewidth=2, zorder=2, label='Upper CI')
    ax.plot(iterations, m, color=color, linewidth=3, zorder=3, label='Mean')

    ax.set_xlabel('Training Iterations')
    ax.set_ylabel(args.metric)
    ax.set_title(f'{args.metric} Over Training Iterations (Mean ± {int(100 * args.confidence)}% CI)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if args.output:
        plt.savefig(os.path.join(args.output, f'{args.metric}_average_ci.pdf'))
    plt.show()
