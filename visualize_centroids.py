#!/usr/bin/env python

import pandas as pd
import sys
import matplotlib.pyplot as plt

def usage():
    print("""
Usage:
  {} data.csv centroidtrace.csv numclusters col1 col2

This creates a 2D plot, where for each point in the datafile (each row), 
one column is used as the X coordinate, and another as the Y coordinate.
The script will also read a centroid trace file, and plot the progression
of the centroids. The starting centroids are marked as a '+', the final
centroids as a 'o'.

""".format(sys.argv[0]))
    sys.exit(-1)

def main():
    try:
        if len(sys.argv) != 6:
            raise Exception("Wrong number of arguments");

        dataFile = sys.argv[1]
        traceFile = sys.argv[2]
        numCenters = int(sys.argv[3])
        col1, col2 = int(sys.argv[4]), int(sys.argv[5])
    except Exception as e:
        print("Error: {}".format(e))
        usage()

    data = pd.read_csv(dataFile, header=None, comment="#").values
    numCols = data.shape[1] 
    colSel = [ False ] * numCols
    colSel[col1] = True
    colSel[col2] = True

    trace = pd.read_csv(traceFile, header=None, comment="#").values
    assert(trace.shape[1] == numCols)

    numIt = trace.shape[0]//numCenters
    trace = trace.reshape((numIt, numCenters, numCols))

    d = data[:,colSel]
    t = trace[:,:,colSel]

    plt.figure()
    plt.plot(d[:,0], d[:,1], '.', color='gray')
    for j in range(numCenters):
        x, y = [ ], [ ]
        for i in range(numIt):
            x.append(float(t[i,j,0]))
            y.append(float(t[i,j,1]))
        plt.plot(x, y, '-o')
    plt.plot(t[0,:,0], t[0,:,1], '+')
    plt.plot(t[-1,:,0], t[-1,:,1], 'o', mfc='none', color='black')
    plt.show()

if __name__ == "__main__":
    main()
