import argparse
import os.path
import re
import yaml

from datetime import datetime
import matplotlib.pyplot as plt


LOG_FILE = 'training.log'
CONFIG_FILE = 'config.yaml'
pattern = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - dqn - INFO - Finished episode \d+; total score (\d+); avg score (\d+); epsilon')


def rolling_mean(data, window):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]


def plot_one(model_dir: str):
    total_scores = []
    avg_scores = []
    timestamps = []

    with open(os.path.join(model_dir, LOG_FILE), 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamps.append(datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f'))
                total_scores.append(int(match.group(2)))
                avg_scores.append(int(match.group(3)))

    rolling_mean_total_scores = rolling_mean(total_scores, 100)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    ax1.plot(avg_scores, label='Average Score')
    ax1.plot(range(99, len(total_scores)), rolling_mean_total_scores, label='Rolling Mean Total Score (100)', linestyle='--')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores per Episode')
    ax1.legend()

    with open(os.path.join(model_dir, CONFIG_FILE), 'r') as f:
        hyperparams = yaml.safe_load(f)

    hyperparams_text = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])
    plt.gcf().text(0.82, 0.10, hyperparams_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5), ha='left',va='bottom')

    plt.tight_layout()
    plt.show()


def plot_multiple(model_dirs: list[str], mean: str):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    for model_dir in model_dirs:
        total_scores = []
        avg_scores = []
        timestamps = []
        model_name = model_dir.split('/')[-1]

        with open(os.path.join(model_dir, LOG_FILE), 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    timestamps.append(datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f'))
                    total_scores.append(int(match.group(2)))
                    avg_scores.append(int(match.group(3)))

        if mean == 'cumulative':
            ax1.plot(avg_scores, label=model_name)
        else:
            rolling_mean_total_scores = rolling_mean(total_scores, 100)
            ax1.plot(range(99, len(total_scores)), rolling_mean_total_scores, label=model_name)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    if mean == 'cumulative':
        ax1.set_title('Cumulative average score at each episode')
    else:
        ax1.set_title('Rolling Mean Total Scores per Episode (100)')
    ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_time():
    models_types = ['ALE_SpaceInvaders-v5-cnn-ram', 'ALE_SpaceInvaders-v5-dense-ram', 'ALE_SpaceInvaders-v5-deep-dense-ram']
    model_dirs = os.listdir('models')
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - dqn - INFO - Finished episode \d+; total score \d+; avg score \d+; epsilon')
    mean_times = {}
    for model_dir in model_dirs:
        with open(os.path.join('models', model_dir, LOG_FILE), 'r') as f:
            timestamps = []
            for line in f:
                match = time_pattern.search(line)
                if match:
                    timestamps.append(datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f'))
            total_time = (timestamps[-1] - timestamps[0]).total_seconds()
            mean_times[model_dir] = total_time / len(timestamps)

    mean_times_type = {}
    for model_type in models_types:
        model_type_times = [mean_times[model_dir] for model_dir in model_dirs if model_dir.startswith(model_type)]
        mean_times_type[model_type] = sum(model_type_times) / len(model_type_times)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.bar(mean_times_type.keys(), mean_times_type.values(), color=colors[:len(mean_times_type)])
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Mean Time (seconds)')
    ax1.set_title('Mean Time per Episode for Each Model Type')
    plt.tight_layout()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--model-dirs', nargs='+')
parser.add_argument('--mean', default='rolling', choices=['rolling', 'cumulative'])
parser.add_argument('--time', action='store_true')

args = parser.parse_args()

if args.time:
    plot_time()
    exit()

path = os.path.split(args.model_dirs[0])
if path[-1] == '*':
    path = os.path.join(*path[:-1])
    args.model_dirs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

if len(args.model_dirs) == 1:
    plot_one(args.model_dirs[0])
else:
    plot_multiple(args.model_dirs, args.mean)