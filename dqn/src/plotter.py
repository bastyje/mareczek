import os.path
import re
import sys
import matplotlib.pyplot as plt
from datetime import datetime

import yaml


def rolling_mean(data, window):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]


pattern = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - dqn - INFO - Finished episode \d+; total score (\d+); avg score (\d+); epsilon')

LOG_FILE = 'training.log'
CONFIG_FILE = 'config.yaml'
model_dir = sys.argv[1]
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

time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0

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
plt.gcf().text(0.82, 0.10, hyperparams_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5), ha='left', va='bottom')

plt.tight_layout()
plt.show()