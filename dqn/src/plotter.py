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


parser = argparse.ArgumentParser()
parser.add_argument('--model-dirs', nargs='+')
parser.add_argument('--mean', default='rolling', choices=['rolling', 'cumulative'])

args = parser.parse_args()

path = os.path.split(args.model_dirs[0])
if [-1] == '*':
    path = os.path.join(*path[:-1])
    args.model_dirs = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

if len(args.model_dirs) == 1:
    plot_one(args.model_dirs[0])
else:
    plot_multiple(args.model_dirs, args.mean)