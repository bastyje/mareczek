import re
import sys
import matplotlib.pyplot as plt


pattern = re.compile(r'Finished episode \d+; total score \d+; avg score (\d+); epsilon')

log_file = sys.argv[1]
scores = []

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            scores.append(int(match.group(1)))

plt.plot(scores)
plt.show()