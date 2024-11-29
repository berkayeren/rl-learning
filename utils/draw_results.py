import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


# Function to flatten nested dictionaries
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Replace this with the path to your results folder
results_folder = 'C:\\Users\\BerkayEren\\Desktop\\dowham 10182024\\results'

# Specify the custom metrics to plot
custom_metric_keys = ['', 'intrinsic_reward_mean', 'percentage_visited_mean', 'done_mean', 'episode_reward_mean']

# Initialize a dictionary to accumulate data
iteration_data = {}

# Traverse through the results folder
for root, dirs, files in os.walk(results_folder):
    for file in files:
        if file == 'result.json':
            result_file_path = os.path.join(root, file)
            print(f"Processing file: {result_file_path}")
            # Open and read the result.json file
            with open(result_file_path, 'r') as f:
                # Read and process the file line by line to handle large files
                for line in f:
                    try:
                        data = json.loads(line)
                        # Extract the necessary data
                        iteration = data.get('training_iteration')
                        if iteration is None:
                            continue
                        iteration = int(iteration)
                        custom_metrics = data.get('custom_metrics', {})
                        # Flatten custom_metrics if needed
                        flat_metrics = flatten_dict(custom_metrics)
                        # Get 'episode_reward_mean' from top-level data
                        episode_reward_mean = data.get('episode_reward_mean', None)
                        if episode_reward_mean is not None:
                            flat_metrics['episode_reward_mean'] = episode_reward_mean
                        # Only keep the specified metrics
                        metric_data = {key: flat_metrics.get(key, None) for key in custom_metric_keys}
                        # Initialize data structure for this iteration if not present
                        if iteration not in iteration_data:
                            iteration_data[iteration] = {key: [] for key in custom_metric_keys}
                        # Append metrics to the lists
                        for key in custom_metric_keys:
                            value = metric_data.get(key)
                            if value is not None:
                                iteration_data[iteration][key].append(value)
                    except json.JSONDecodeError:
                        print(f"Skipping corrupted line in {result_file_path}")
                        continue

# Now, compute mean of metrics per iteration
data_list = []
for iteration in sorted(iteration_data.keys()):
    data_point = {'training_iteration': iteration}
    for key in custom_metric_keys:
        values = iteration_data[iteration][key]
        if values:
            data_point[key] = sum(values) / len(values)
        else:
            data_point[key] = None
    data_list.append(data_point)

# Convert the list of dicts to a pandas DataFrame
df = pd.DataFrame(data_list)

# Remove any columns with all NaN values (optional)
df.dropna(axis=1, how='all', inplace=True)

# Drop rows with NaN in metrics
df.dropna(subset=custom_metric_keys, how='all', inplace=True)

# Set the index to training_iteration
df.set_index('training_iteration', inplace=True)

# Plot each custom metric with LOWESS smoothing
for metric in custom_metric_keys:
    if metric in df.columns:
        plt.figure()
        x = df.index.values
        y = df[metric].values

        # Apply LOWESS smoothing
        frac = 0.05  # Adjust the fraction as needed
        y_smoothed = lowess(y, x, frac=frac, return_sorted=False)

        plt.plot(x, y, label='Original Data', alpha=0.3, linewidth=0.5)
        plt.plot(x, y_smoothed, 'r-', label='Smoothed Curve')

        plt.title(f'{metric} over Training Iterations')
        plt.xlabel('Training Iteration')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"Metric '{metric}' not found in the data.")
