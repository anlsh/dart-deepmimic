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
        # data = f.read()
        matches = re.findall(pattern, data)

    # data = np.zeros([3, len(matches)])
    # for index, match in enumerate(matches):
    #     data[i] = np.array([float(match.group(m)) for m in range(1, 3)])
    data = np.array(matches).astype(np.float).T

    # print(args.typed)
    # plt.plot(data[0], data[1 if args.typed else 2])
    # plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(data[3], data[1])
    plt.title('A tale of 2 subplots')
    plt.ylabel('Avg Episode Length')

    plt.subplot(2, 1, 2)
    plt.plot(data[3], data[2])
    plt.ylabel('Avg Episode Reward')
    plt.xlabel('Number of "iterations"')

    plt.show()
