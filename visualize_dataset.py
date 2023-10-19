#!/usr/bin/env python

import pandas as pd
import sys
import matplotlib.pyplot as plt
import pprint
import numpy as np

def usage():
    print("""
Usage:
  {} data.csv col1 col2

This creates a 2D plot, where for each point in the datafile (each row), 
one column is used as the X coordinate, and another as the Y coordinate.

""".format(sys.argv[0]))
    sys.exit(-1)

def main():
    try:
        if len(sys.argv) != 4:
            raise Exception("Wrong number of arguments");

        dataFile = sys.argv[1]
        col1, col2 = int(sys.argv[2]), int(sys.argv[3])
    except Exception as e:
        print("Error: {}".format(e))
        usage()

    data = pd.read_csv(dataFile, header=None, comment="#").values

    t = "Matrix has {} rows and {} columns".format(data.shape[0], data.shape[1])
    print(t)
    plt.title(t)
    plt.plot(data[:,col1], data[:,col2], '.')
    plt.show()

if __name__ == "__main__":
    main()
