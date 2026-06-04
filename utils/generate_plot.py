import os
import os.path

import matplotlib as mpl
import numpy as np

mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from cycler import cycler
from matplotlib.font_manager import FontProperties
import progressbar
import subprocess
import pandas as pd
import json
import argparse

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica"]


# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{dutchcal} \usepackage{amssymb}')

def smooth_values(y):
    return savgol_filter(y, window_length=31, polyorder=3)


def print_keys(obj, indent):
    if isinstance(obj, dict):
        for key in obj.keys():
            print(" " * indent + f"- {key} : {obj[key]}")
            print_keys(obj[key], indent + 2)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(" " * indent + f"- [{i}]")
            print_keys(item, indent + 2)


def get_expanded_metric_list(metric):
    return [metric, "{}_min".format(metric), "{}_mean".format(metric), "{}_max".format(metric)]


def get_plot_config(trial_directory):
    result_directory = os.path.join(os.path.expanduser('~'), "ray_results")

    json_path = os.path.join(result_directory, trial_directory, 'plot_config.json')
    print(json_path)
    if not os.path.exists(json_path):
        return None

    f = open(json_path, 'r')
    return json.load(f)


def get_dataframe(trial_directory, metric):
    result_directory = os.path.join(os.path.expanduser('~'), "ray_results")

    columns = ["training_iteration", "timesteps_total"] + get_expanded_metric_list(metric)
    json_path = os.path.join(result_directory, trial_directory, 'result.json')
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        lines = f.readlines()
        number_of_iterations = len(lines)
        value_df = pd.DataFrame(np.nan, index=range(number_of_iterations), columns=columns)

        print("Parsing JSON file...")
        bar = progressbar.ProgressBar(maxval=number_of_iterations,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        metrics_exists = False
        for l in range(len(lines)):
            line_data = json.loads(lines[l])
            if l == 0:
                print(line_data['config']['env'])
                print(line_data['config'])
                print_keys(line_data['config']['env_config'], 1)

            training_iteration = line_data['training_iteration']
            timesteps_total = line_data['timesteps_total']
            value_df.iloc[training_iteration - 1]['training_iteration'] = training_iteration
            value_df.iloc[training_iteration - 1]['timesteps_total'] = timesteps_total

            metric_dir = metric.split('/')
            is_metric_found = True

            for d in metric_dir[:-1]:
                if d not in line_data.keys():
                    is_metric_found = False
                    break
                line_data = line_data[d]

            if is_metric_found:
                for m in get_expanded_metric_list(metric_dir[-1]):
                    if m in line_data.keys():
                        value_df.iloc[training_iteration - 1]['/'.join(metric_dir[:-1] + [m])] = line_data[m]
                        metrics_exists = True
                    elif m in line_data['custom_metrics']:
                        value_df.iloc[training_iteration - 1]['/'.join(metric_dir[:-1] + [m])] = \
                            line_data['custom_metrics'][m]
                        metrics_exists = True
            bar.update(l + 1)
        bar.finish()

        if not metrics_exists:
            print("Metric is not found. Here is the structure of evaluation key in the JSON line:")
            line_data = json.loads(lines[0])
            evaluation_interval = line_data["config"]["evaluation_interval"]
            print_keys(json.loads(lines[evaluation_interval - 1])[metric.split('/')[0]], 0)
            return None

        value_df["training_iteration"] = value_df["training_iteration"].astype(int)
        value_df["timesteps_total"] = value_df["timesteps_total"].astype(int)
        value_df.dropna(axis=1, how='all', inplace=True)
        value_df.dropna(inplace=True)
        return value_df
    return None


def plot_experiment(ax, df, metric, trial_directory, legend_name, plot_config, error_bounds=True, smooth=False):
    if plot_config:
        ln = plot_config['legend_name']
        lw = float(plot_config['linewidth'])
        c = plot_config['color']
        ls = plot_config['linestyle']
        m = plot_config['marker']
        if error_bounds:
            if smooth:
                ax.fill_between(df['timesteps_total'], smooth_values(df["{}_min".format(metric)]),
                                smooth_values(df["{}_max".format(metric)]), alpha=0.2, facecolor=c)
                ax.plot(df['timesteps_total'], smooth_values(df["{}_mean".format(metric)]), label=ln, linewidth=lw,
                        color=c, linestyle=ls, marker=m)
            else:
                ax.fill_between(df['timesteps_total'], df["{}_min".format(metric)], df["{}_max".format(metric)],
                                alpha=0.2, facecolor=c)
                ax.plot(df['timesteps_total'], df["{}_mean".format(metric)], label=ln, linewidth=lw, color=c,
                        linestyle=ls, marker=m)
        else:
            if smooth:
                ax.plot(df['timesteps_total'], smooth_values(df[metric]), label=ln, linewidth=lw, color=c, linestyle=ls,
                        marker=m)
            else:
                ax.plot(df['timesteps_total'], df[metric], label=ln, linewidth=lw, color=c, linestyle=ls, marker=m)
    else:
        if error_bounds:
            if smooth:
                ax.fill_between(df['timesteps_total'], smooth_values(df["{}_min".format(metric)]),
                                smooth_values(df["{}_max".format(metric)]), alpha=0.2)
                ax.plot(df['timesteps_total'], smooth_values(df["{}_mean".format(metric)]), label=legend_name,
                        linewidth=2.6)
            else:
                try:
                    ax.fill_between(df['timesteps_total'], df["{}_min".format(metric)], df["{}_max".format(metric)],
                                    alpha=0.2)
                except KeyError:
                    print("Error bounds are not available")
                ax.plot(df['timesteps_total'], df["{}_mean".format(metric)], label=legend_name, linewidth=2.6)
        else:
            if smooth:
                ax.plot(df['timesteps_total'], smooth_values(df[metric]), label=legend_name, linewidth=2.6)
            else:
                ax.plot(df['timesteps_total'], df[metric], label=legend_name, linewidth=2.6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("metric")
    parser.add_argument("-ymin", "--ymin", default=0, help="lower bound on Y axis")
    parser.add_argument("-ymax", "--ymax", default=0, help="upper bound on Y axis")
    parser.add_argument("-xmin", "--xmin", default=0, help="lower bound on X axis")
    parser.add_argument("-xmax", "--xmax", default=0, help="upper bound on X axis")
    parser.add_argument("-me", "--multiexp", default=False, help="if there are multiple experiments to plot")
    parser.add_argument("-us", "--undersampling", default=1, help="undersampling")
    parser.add_argument("-sm", "--smoothing", default=False, help="smoothing")
    args = vars(parser.parse_args())

    trial_directory = args['directory']
    metric = args['metric']
    y_min = float(args['ymin'])
    y_max = float(args['ymax'])
    x_min = float(args['xmin'])
    x_max = float(args['xmax'])
    us = int(args['undersampling'])

    result_directory = os.path.expanduser('~') + "/ray_results"

    mpl.rcParams.update({'font.size': 36})
    plt.rc('lines', linewidth=4)

    # main plots
    plt.rc('axes', prop_cycle=(
            cycler('color', ['#e41a1c', '#4daf4a', '#1f78b4', '#cd00cc', '#ff8000', '#00D8F9', '#984ea3']) + cycler(
        'linestyle', ['-', '-', '-', '-', '-', '-', '-'])))

    fig = plt.figure(figsize=(32, 14), dpi=200)
    ax = fig.add_subplot(111)
    exp_dirs = []
    if args['multiexp']:
        full_directory = os.path.join(os.path.expanduser('~'), "ray_results", trial_directory)
        dirs = os.listdir(full_directory)

        for d in dirs:
            if os.path.isdir(os.path.join(full_directory, d)):
                exp_dirs.append(os.path.join(trial_directory, d))
    else:
        exp_dirs.append(trial_directory)

    x_values = None
    for ed in exp_dirs:
        print("\nReading", ed)
        value_df = get_dataframe(ed, metric)
        plot_config = get_plot_config(ed)

        if isinstance(value_df, pd.DataFrame):
            df_x_values = np.asarray(value_df['timesteps_total'])
            if x_values is None or (isinstance(x_values, np.ndarray) and df_x_values.shape[0] > x_values.shape[0]):
                x_values = df_x_values

            print(value_df.iloc[-100:])
            value_df = value_df.iloc[::us]

            plot_experiment(ax, value_df, metric, trial_directory, ed, plot_config, smooth=args['smoothing'])
            print(value_df[["training_iteration", "timesteps_total"]].max())
            print("====== Average of Last 100 Samples ======")
            metric_df = value_df.drop(["training_iteration", "timesteps_total"], axis=1)
            print(metric_df.iloc[-100:].mean())
        else:
            print("Cannot read values")

    only_metric_name = metric.split('/')[-1]
    ax.set(xlabel='Total Time Steps', ylabel=only_metric_name)

    fontP = FontProperties()
    fontP.set_size(32)

    ax.legend(bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=3, columnspacing=0.8, handletextpad=0.3, prop=fontP)
    ax.grid(axis='y', color='lightgray', linestyle='dotted', linewidth=1)

    if y_min != y_max:
        plt.ylim(y_min, y_max)

    if x_min != x_max:
        plt.xlim(x_min, x_max)

    trial_dir = os.path.join(result_directory, trial_directory)
    pdf_file_name = "{}/plot_{}.pdf".format(trial_dir, only_metric_name)
    fig.savefig(pdf_file_name, format="pdf")
