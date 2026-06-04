import argparse
import os
import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
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


def scipy_bootstrap_confidence_interval(data, confidence_level=0.95, n_resamples=1000):
    """Calculate bootstrap confidence interval using scipy.stats.bootstrap."""
    if len(data) <= 1:
        return np.mean(data), (np.mean(data), np.mean(data))

    # Scipy bootstrap expects data as a tuple of arrays
    data_tuple = (np.array(data),)

    # Define the statistic function (mean)
    def mean_statistic(x):
        return np.mean(x)

    # Perform bootstrap
    res = bootstrap(data_tuple, mean_statistic, n_resamples=n_resamples,
                    confidence_level=confidence_level, method='percentile',
                    random_state=42)

    return np.mean(data), (res.confidence_interval.low, res.confidence_interval.high)


def calculate_bootstrap_ci_for_metrics(metrics_data, n_resamples=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics using scipy."""
    results = {}

    for exp_name, df in metrics_data.items():
        metric_values = df.iloc[:, 1].values  # Metric column is the second one
        mean_val, (ci_lower, ci_upper) = scipy_bootstrap_confidence_interval(
            metric_values,
            confidence_level=confidence_level,
            n_resamples=n_resamples
        )

        results[exp_name] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    return results


def calculate_confidence_intervals_over_time(metrics_data, bin_size=None, method='bootstrap',
                                             n_resamples=1000, confidence_level=0.95):
    """
    Calculate confidence intervals for metrics at each iteration or binned iterations using scipy.

    Args:
        metrics_data: Dictionary of dataframes with iterations and metric values
        bin_size: If provided, bin iterations into groups of this size
        method: Method to calculate confidence intervals ('bootstrap' or 'std_error')
        n_resamples: Number of bootstrap resamples if using bootstrap method
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
        for i, vals in enumerate(tqdm(values, desc=f"Computing CI for {exp_name}")):
            if len(vals) <= 1:
                # Not enough data for CI calculation
                lower_bounds[i] = means[i]
                upper_bounds[i] = means[i]
                continue

            if method == 'bootstrap':
                # Use scipy bootstrap method
                mean_val, (lb, ub) = scipy_bootstrap_confidence_interval(
                    vals, confidence_level=confidence_level, n_resamples=n_resamples
                )
                lower_bounds[i] = lb
                upper_bounds[i] = ub
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
                                                     n_resamples=1000,
                                                     confidence_level=0.95):
    """Compute overall mean ± bootstrap CI across runs at each iteration/bin using scipy."""
    # 1) Extract each run's mean-series
    run_means = {}
    for exp_name, df in metrics_data.items():
        # df now has columns ['iteration','mean','lower_bound','upper_bound']
        s = df.set_index('iteration')['mean']
        run_means[exp_name] = s

    # 2) Align on union of all iterations, then fill missing per-run means
    all_iterations = sorted(set().union(*(s.index for s in run_means.values())))
    runs_df = pd.DataFrame(
        {exp: run_means[exp].reindex(all_iterations) for exp in run_means},
        index=all_iterations
    ).sort_index().ffill().bfill()

    # 3) Bootstrap across runs at each iteration using scipy
    means, lower_bounds, upper_bounds = [], [], []
    for vals in tqdm(runs_df.values, desc="Computing average CIs"):
        mean_val, (lb, ub) = scipy_bootstrap_confidence_interval(
            vals,
            confidence_level=confidence_level,
            n_resamples=n_resamples
        )
        means.append(mean_val)
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    return pd.DataFrame({
        'iteration': all_iterations,
        'mean': means,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })


def smooth_data(data, window_size=5, method='moving_average'):
    """
    Apply smoothing to data.

    Args:
        data: Array-like data to smooth
        window_size: Size of the smoothing window
        method: Smoothing method ('moving_average', 'exponential')

    Returns:
        Smoothed data array
    """
    if window_size <= 1 or len(data) <= window_size:
        return data

    if method == 'moving_average':
        # Simple moving average
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        # Pad the beginning to maintain original length
        padding = np.full(window_size - 1, smoothed[0])
        return np.concatenate([padding, smoothed])

    elif method == 'exponential':
        # Exponential moving average
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    else:
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate bootstrap confidence intervals for Ray experiments using scipy")
    parser.add_argument("--exp_dirs", nargs='+', required=True, help="List of experiment directories")
    parser.add_argument("--metric", required=True, help="Metric name to analyze")
    parser.add_argument("--n_resamples", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--bin_size", type=int, help="Bin size for iteration grouping (optional)")
    parser.add_argument("--ci_method", choices=['bootstrap', 'std_error'], default='bootstrap',
                        help="Method to calculate confidence intervals")
    parser.add_argument("--sqlite_db", help="Path to SQLite database file to store results (optional)")
    parser.add_argument("--min_timesteps", type=int, default=50_000_000,
                        help="Minimum timesteps to load from result.json (default: 50M)")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Window size for smoothing (1 = no smoothing)")
    parser.add_argument("--smooth_method", choices=['moving_average', 'exponential'],
                        default='moving_average', help="Smoothing method")
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

    # Optionally write results to SQLite
    if args.sqlite_db:
        conn = sqlite3.connect(args.sqlite_db)
        all_rows = []
        for exp_name, df in metrics_data.items():
            # Add metric name as a column
            df = df.copy()
            df['metric'] = args.metric
            df['experiment'] = exp_name
            # Seed is already in the dataframe from extract_evaluation_metrics
            all_rows.append(df)

        if all_rows:
            all_df = pd.concat(all_rows, ignore_index=True)
            all_df.to_sql('experiment_results', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Results written to SQLite database: {args.sqlite_db}")

    # 1) Compute per-experiment CI over time (fills mean, lower_bound, upper_bound)
    metrics_data = calculate_confidence_intervals_over_time(
        metrics_data,
        bin_size=args.bin_size,
        method=args.ci_method,
        n_resamples=args.n_resamples,
        confidence_level=args.confidence
    )

    # 2) Compute overall average ± CI across those run-means
    avg_df = calculate_average_confidence_intervals_over_time(
        metrics_data,
        bin_size=None,  # already binned above if needed
        n_resamples=args.n_resamples,
        confidence_level=args.confidence
    )

    # 3) Plot single mean curve + its bootstrap CI
    fig, ax = plt.subplots(figsize=(12, 8))
    iterations = avg_df['iteration']
    m = avg_df['mean']
    lb = avg_df['lower_bound']
    ub = avg_df['upper_bound']

    # Apply smoothing if requested
    if args.smooth_window > 1:
        m_smooth = smooth_data(m, args.smooth_window, args.smooth_method)
        lb_smooth = smooth_data(lb, args.smooth_window, args.smooth_method)
        ub_smooth = smooth_data(ub, args.smooth_window, args.smooth_method)

        # Plot original data with transparency
        ax.fill_between(iterations, lb, ub, color='lightgray', alpha=0.3,
                        label=f'{int(100 * args.confidence)}% CI (raw)', zorder=0)
        ax.plot(iterations, m, color='gray', alpha=0.4, linewidth=1,
                label='Mean (raw)', zorder=1)

        # Plot smoothed data
        ax.fill_between(iterations, lb_smooth, ub_smooth, color='#aec7e8', alpha=0.5,
                        label=f'{int(100 * args.confidence)}% CI (smooth)', zorder=2)
        ax.plot(iterations, lb_smooth, linestyle='--', color='#4a90e2', linewidth=2,
                zorder=3, label='Lower CI (smooth)')
        ax.plot(iterations, ub_smooth, linestyle='--', color='#4a90e2', linewidth=2,
                zorder=3, label='Upper CI (smooth)')
        ax.plot(iterations, m_smooth, color='#1f77b4', linewidth=3, zorder=4,
                label=f'Mean (smooth, window={args.smooth_window})')
    else:
        # Use original color scheme without smoothing
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

    title = f'{args.metric} Over Training Iterations (Mean ± {int(100 * args.confidence)}% CI)'
    if args.smooth_window > 1:
        title += f' - Smoothed (window={args.smooth_window})'
    ax.set_title(title)

    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if args.output:
        filename = f'{args.metric}_average_ci'
        if args.smooth_window > 1:
            filename += f'_smooth{args.smooth_window}'
        plt.savefig(os.path.join(args.output, f'{filename}.pdf'))
    plt.show()
    exit()
