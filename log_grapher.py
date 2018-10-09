# Iteration ([0-9]+) *.*?EpLenMean\s+\|\s+([0-9\.]+).*?EpRewMean\s+\|\s+([0-9\.]+)

import matplotlib.pyplot as plt
import re
import mmap
import numpy as np
import sys

if __name__ == "__main__":

    pattern = re.compile(b"Iteration ([0-9]+) *.*?EpLenMean\s+\|\s+([0-9\.]+).*?EpRewMean\s+\|\s+([0-9\.]+).*?TimestepsSoFar\s+\|\s+([0-9e\+\.]+)",
                         re.M | re.S)

    with open(sys.argv[1], "r") as f:
        data = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        matches = re.findall(pattern, data)

    data = np.array(matches).astype(np.float).T

    plt.subplot(3, 1, 1)
    plt.plot(data[3], data[1])
    plt.title('A tale of 2 subplots')
    plt.ylabel('Avg Episode Length')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.subplot(3, 1, 2)
    plt.plot(data[3], data[2])
    plt.ylabel('Avg cumulative reward')
    plt.xlabel('Timesteps of experience collected"')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.subplot(3, 1, 3)
    avg_reward_data = np.divide(data[2], data[1])
    plt.plot(data[3], avg_reward_data)
    plt.ylabel('Avg reward per timestep')
    plt.xlabel('Timesteps of experiene collected')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.show()
