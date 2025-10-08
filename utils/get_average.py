import ast
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file

if __name__ == "__main__":
    # Print the DataFrame
    states = {}
    total_steps = 0
    for i in range(0, 10):
        df = pd.read_csv(f'trail-50k-{i}.csv', names=['Key', 'Value'])

        for index, row in df.iterrows():
            key = ast.literal_eval(row['Key'])
            value = int(row['Value'])
            states.setdefault(key, 0)
            states[key] += value
            total_steps += value
    # Find the minimum and maximum values in the states dictionary
    min_value = min(states.values())
    max_value = max(states.values())

    # Normalize the values in the states dictionary
    for key in states:
        states[key] = round(float((states[key] - min_value) / (max_value - min_value)), 3)

    visited_states_array = np.zeros((25, 25))

    # Iterate over the visited states
    for position, count in states.items():
        # Set the value at the agent's position in the array to the visit count
        visited_states_array[position] = count

    plt.imshow(visited_states_array, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.scatter(14, 7, c='green', marker='*', s=800)

    plt.title(f'Heatmap of Visited States in Trial {i + 1}')

    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    # Save the heatmap to a file
    plt.savefig(
        f'C:\\Users\\BerkayEren\\PycharmProjects\\rl-learning\\heatmaps\\heatmap_dowham_trial_50k.png')

    # Clear the current figure so the next heatmap doesn't overlap with this one
    plt.clf()
