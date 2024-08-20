import json
import os

if __name__ == "__main__":
    # Directory path
    base_path = r"C:\Users\BerkayEren\PycharmProjects\rl-learning\new_rl_results"

    # Get list of all directories
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


    # Function to extract the date from directory name
    def get_date_from_directory_name(directory_name):
        try:
            date_str = directory_name.split('_')[2]
            return datetime.strptime(date_str, '%Y-%m-%d')
        except (IndexError, ValueError):
            return None


    # Create a list of tuples (directory, date)
    dir_date_list = [(d, get_date_from_directory_name(d)) for d in directories]

    # Filter out directories that don't have a valid date
    dir_date_list = [d for d in dir_date_list if d[1] is not None]

    # Sort directories by date
    sorted_directories = sorted(dir_date_list, key=lambda x: x[1])


    # Function to process the large JSON file incrementally
    def process_large_json(file_path):
        extracted_data = []
        with open(file_path, 'r') as f:
            for line in f:
                # You can customize the processing here
                try:
                    data = json.loads(line.strip())
                    # Extract specific information or summarize the data
                    extracted_data.append(data)  # Modify this line to extract necessary info
                except json.JSONDecodeError:
                    continue
        return extracted_data


    def show_metrics(results, key, kind='linear', window_size=101, color='tab:blue'):
        # Sort the results by timestamp
        sorted_results = sorted(results, key=lambda x: x['timestamp'])

        # Extract the relevant metric
        metrics = [entry[key] for entry in sorted_results]

        # Create an index for plotting
        indices = list(range(len(metrics)))

        # Interpolate the data
        interpolator = interp1d(indices, metrics, kind=kind)
        smooth_indices = np.linspace(0, len(metrics) - 1, num=5000)
        smooth_metrics = interpolator(smooth_indices)

        # Apply moving average for additional smoothing
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        smoothed_metrics = moving_average(smooth_metrics, window_size)
        adjusted_indices = smooth_indices[
                           (window_size - 1) // 2: -(window_size - 1) // 2]  # Adjust indices for moving average

        # Plotting the data
        plt.figure(figsize=(20, 10))
        plt.plot(adjusted_indices, smoothed_metrics, linestyle='-', label=key, color=color)
        plt.xlabel('Index')
        plt.ylabel(key)
        plt.title(f'Progress of {key} Over Time')
        plt.grid(True)
        plt.show()


    # Read and process result.json files
    intrinsic_reward_results = []
    count_based_results = []

    for directory, date in sorted_directories:
        result_file_path = os.path.join(base_path, directory, 'result.json')
        if os.path.exists(result_file_path):
            result_data = process_large_json(result_file_path)
            for data in result_data:
                new_data = {}
                if 'percentage_visited_mean' in data['custom_metrics']:
                    new_data['percentage_visited_mean'] = data['custom_metrics']['percentage_visited_mean']
                    new_data['episode_reward_mean'] = data['episode_reward_mean']
                    new_data['timestamp'] = data['timestamp']

                if 'intrinsic_reward_mean' in data['custom_metrics']:
                    new_data['intrinsic_reward_mean'] = data['custom_metrics']['intrinsic_reward_mean']
                    intrinsic_reward_results.append(new_data)
                    continue

                if 'count_bonus_mean' in data['custom_metrics']:
                    new_data['intrinsic_reward_mean'] = data['custom_metrics']['count_bonus_mean']
                    count_based_results.append(new_data)

    show_metrics(intrinsic_reward_results, 'intrinsic_reward_mean', kind='quadratic', window_size=101)
    show_metrics(intrinsic_reward_results, 'percentage_visited_mean', kind='quadratic', window_size=101)
    show_metrics(intrinsic_reward_results, 'episode_reward_mean', kind='quadratic', window_size=101)

    show_metrics(count_based_results, 'intrinsic_reward_mean', kind='quadratic', window_size=101, color='orange')
    show_metrics(count_based_results, 'percentage_visited_mean', kind='quadratic', window_size=101, color='orange')
    show_metrics(count_based_results, 'episode_reward_mean', kind='quadratic', window_size=101, color='orange')

    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    from scipy.interpolate import interp1d


    def show_metrics_gather(datasets, key, labels, kind='linear', window_size=101):
        plt.figure(figsize=(20, 10))

        for i, results in enumerate(datasets):
            # Sort the results by timestamp
            sorted_results = sorted(results, key=lambda x: x['timestamp'])

            # Extract the relevant metric
            metrics = [entry[key] for entry in sorted_results]

            # Create an index for plotting
            indices = list(range(len(metrics)))

            # Interpolate the data
            interpolator = interp1d(indices, metrics, kind=kind)
            smooth_indices = np.linspace(0, len(metrics) - 1, num=5000)
            smooth_metrics = interpolator(smooth_indices)

            # Apply moving average for additional smoothing
            def moving_average(data, window_size):
                return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

            smoothed_metrics = moving_average(smooth_metrics, window_size)
            adjusted_indices = smooth_indices[
                               (window_size - 1) // 2: -(window_size - 1) // 2]  # Adjust indices for moving average

            # Plotting the data
            plt.plot(adjusted_indices, smoothed_metrics, linestyle='-', label=labels[i])

        plt.xlabel('Index')
        plt.ylabel(key)
        plt.title(f'Progress of {key} Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Call the function with the desired metrics and labels
    # datasets = [intrinsic_reward_results, count_based_results]
    # labels = ['Intrinsic Reward Results', 'Count Based Results']
    # show_metrics_gather(datasets, 'intrinsic_reward_mean', labels)
    # show_metrics_gather(datasets, 'percentage_visited_mean', labels)
    # show_metrics_gather(datasets, 'episode_reward_mean', labels)
