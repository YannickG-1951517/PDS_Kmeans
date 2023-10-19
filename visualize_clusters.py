#!/usr/bin/env python

import pandas as pd
import sys
import matplotlib.pyplot as plt
import pprint
import numpy as np

def usage():
    print("""
Usage:
  {} data.csv clustertrace.csv col1 col2

For each line in the cluster trace file (or output file), this creates a 2D
plot where for each point in the datafile (each row), one column is used as
the X coordinate, and another as the Y coordinate. Points belonging to
different clusters will have a different color.

""".format(sys.argv[0]))
    sys.exit(-1)

def main():
    try:
        if len(sys.argv) != 5:
            raise Exception("Wrong number of arguments");

        dataFile = sys.argv[1]
        clusterFile = sys.argv[2]
        col1, col2 = int(sys.argv[3]), int(sys.argv[4])
    except Exception as e:
        print("Error: {}".format(e))
        usage()

    data = pd.read_csv(dataFile, header=None, comment="#").values

    for clusterLine in open(clusterFile, "rt").readlines():
        if clusterLine.startswith("#"):
            continue
        clusters = np.array(list(map(int, clusterLine.split(","))))

        numClusters = clusters.max() + 1

        plt.figure()
        for c in range(numClusters):
            d = data[clusters==c]
            plt.plot(d[:,col1], d[:,col2], '.')
        plt.show()

if __name__ == "__main__":
    main()
