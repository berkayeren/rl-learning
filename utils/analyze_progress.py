import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt


def summarize_data(df):
    """Display summary statistics, data types, and missing values."""
    print("Summary Statistics:")
    print(df.describe(include='all'))
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())


def clean_and_process_data(df):
    """Clean and process the DataFrame."""
    # Replace 'nan' with NaN for processing
    df.replace('nan', np.nan, inplace=True)

    # Convert appropriate columns to numeric
    numeric_columns = [
        "episode_reward_max", "episode_reward_min", "episode_reward_mean",
        "episode_len_mean", "training_iteration"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values with 0 for simplicity in this example
    df.fillna(0, inplace=True)

    return df


def analyze_and_plot(df):
    """Analyze the processed data and plot key metrics."""
    # Ensure the relevant columns are numeric
    columns_to_plot = [
        "episode_reward_max", "episode_reward_min",
        "episode_reward_mean",
    ]
    for col in columns_to_plot:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Generate the plot
    plt.figure(figsize=(12, 6))
    for col in columns_to_plot:
        plt.plot(df["training_iteration"], df[col], label=col)

    plt.title("Training Progress Metrics")
    plt.xlabel("Training Iteration")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a CSV file and analyze its data.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to process.")

    # Parse arguments
    args = parser.parse_args()

    try:
        # Read CSV file
        data = pd.read_csv(args.csv_file)
        print("Original Data:")
        print(data.head())

        # Summarize, clean, and analyze the data
        summarize_data(data)
        cleaned_data = clean_and_process_data(data)
        analyze_and_plot(cleaned_data)

    except Exception as e:
        print(f"Error processing the file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
