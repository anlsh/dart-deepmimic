# Iteration ([0-9]+) *.*?EpLenMean\s+\|\s+([0-9\.]+).*?EpRewMean\s+\|\s+([0-9\.]+)

import argparse
import matplotlib.pyplot as plt
import re
import mmap
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, default=None,
                        help="The log file you want to parse")
    # type_group = parser.add_mutually_exclusive_group()
    # type_group.add_argument('--reward',
    #                         dest='typed',
    #                         action='store_true')
    # type_group.add_argument('--length',
    #                         dest='typed',
    #                         action='store_false')
    # parser.set_defaults(typed=True, help="What data to plot")

    args = parser.parse_args()

    pattern = re.compile(b"Iteration ([0-9]+) *.*?EpLenMean\s+\|\s+([0-9\.]+).*?EpRewMean\s+\|\s+([0-9\.]+)",
                         re.M | re.S)

    with open(args.log_file, "r") as f:
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
    plt.plot(data[0], data[1])
    plt.title('A tale of 2 subplots')
    plt.ylabel('Avg Episode Length')

    plt.subplot(2, 1, 2)
    plt.plot(data[0], data[2])
    plt.ylabel('Avg Episode Reward')
    plt.xlabel('Number of "iterations"')

    plt.show()
