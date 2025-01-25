import re
import sys
import matplotlib.pyplot as plt
from datetime import datetime


def rolling_mean(data, window):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window + 1)]


pattern = re.compile(
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - dqn - INFO - Finished episode \d+; total score (\d+); avg score (\d+); epsilon')

log_file = sys.argv[1]
total_scores = []
avg_scores = []
timestamps = []

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            timestamps.append(datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f'))
            total_scores.append(int(match.group(2)))
            avg_scores.append(int(match.group(3)))

rolling_mean_total_scores = rolling_mean(total_scores, 100)

time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
avg_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ax1.plot(total_scores, label='Total Score')
ax1.plot(avg_scores, label='Average Score')
ax1.plot(range(99, len(total_scores)), rolling_mean_total_scores, label='Rolling Mean Total Score (100)', linestyle='--')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.set_title('Scores per Episode')
ax1.legend()

ax2.plot([0] + time_diffs, label='Episode Time')
ax2.axhline(y=avg_time, color='r', linestyle='--', label=f'Average Time: {avg_time:.2f} seconds')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Time per Episode')
ax2.legend()

plt.tight_layout()
plt.show()