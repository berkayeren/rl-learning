import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ndjson_to_df(trial_directory, metric):
    json_path = os.path.join(trial_directory, 'result.json')
    records = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Make sure the line is not empty
                # Parse line as JSON
                record = json.loads(line)
                if 'evaluation' in record:
                    record['evaluation']['timesteps_total'] = record['timesteps_total']
                    record['evaluation']['training_iteration'] = record['training_iteration']
                    is_metric_found = False
                    values = {}
                    try:
                        values = {
                            'timesteps_total': record['timesteps_total'],
                            'training_iteration': record['training_iteration'],
                            'episode_len_mean': record['evaluation']['episode_len_mean'],
                            metric + '_min': record['evaluation'][metric + '_min'],
                            metric + '_max': record['evaluation'][metric + '_max'],
                            metric + '_mean': record['evaluation'][metric + '_mean']
                        }
                        is_metric_found = True
                    except KeyError:
                        values = {
                            'timesteps_total': record['timesteps_total'],
                            'training_iteration': record['training_iteration'],
                            'episode_len_mean': record['evaluation']['episode_len_mean'],
                            metric + '_min': record['evaluation']['custom_metrics'][metric + '_min'],
                            metric + '_max': record['evaluation']['custom_metrics'][metric + '_max'],
                            metric + '_mean': record['evaluation']['custom_metrics'][metric + '_mean']
                        }
                        is_metric_found = True

                    records.append(values)
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("metric")
    args = vars(parser.parse_args())
    trial_directory = args['directory']
    metric = args['metric']

    # Load the JSON into a DataFrame
    value_df = load_ndjson_to_df(trial_directory, metric)

    # Construct the column names based on the given metric
    metric_min_col = f"{metric}_min"
    metric_max_col = f"{metric}_max"
    metric_mean_col = f"{metric}_mean"

    # Check if the required columns are present
    required_cols = [metric_min_col, metric_max_col, metric_mean_col, 'timesteps_total']
    if not all(col in value_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in value_df.columns]
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    df_x_values = np.asarray(value_df['timesteps_total'])
    df_min_values = np.asarray(value_df[metric_min_col])
    df_max_values = np.asarray(value_df[metric_max_col])
    df_mean_values = np.asarray(value_df[metric_mean_col])

    # Plot the mean line
    plt.plot(df_x_values, df_mean_values, label=f'{metric}_mean', color='blue')

    # Shade the area between min and max
    plt.fill_between(df_x_values, df_min_values, df_max_values, color='blue', alpha=0.2, label=f'{metric}_range')

    # Compute rolling mean (with a window of 100 data points)
    rolling_mean = value_df[metric_mean_col].rolling(window=100, min_periods=1).mean()
    # Convert rolling mean to numpy array for plotting
    df_rolling_mean = rolling_mean.to_numpy()

    # Plot the rolling mean as a separate line to show the trend
    plt.plot(df_x_values, df_rolling_mean, label=f'{metric}_rolling_mean_100', color='red', linestyle='--', marker='.')

    metric_df = value_df.drop(["training_iteration", "timesteps_total"], axis=1)
    print(metric_df.mean())

    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel(metric)
    plt.title(f'{metric} over Timesteps (Min/Mean/Max)')
    plt.show()
