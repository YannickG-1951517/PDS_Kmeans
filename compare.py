#!/usr/bin/env python
import sys
import os
import subprocess

def usage():
    print("""
compare.py kmeans1|SKIP kmeans2|SKIP --input inputfile.csv --output outputfile.csv --k numclusters --repetitions numrepetitions --seed seed [--threads numthread] [--blocks numblocks] [--trace clusteridxdebug.csv] [--centroidtrace centroiddebug.csv]

This runs two kmeans executables, and compares their output files. The rest
of the arguments are passed on to the executables themselves, but with a
slight modification: the output filenames for the first executable get an
extra '.1' suffix, those for the second executable get '.2'.

If an executable name is 'SKIP', the output files for that run are assumed
to exist already and no program is executed. This can save some time when
developing/testing a new program, to compare it multiple times to the same
reference output.

""", file=sys.stderr)
    sys.exit(-1)

def diffCommand(f1, f2):
    subprocess.check_call(["diff", f1, f2], stdout=subprocess.DEVNULL)

def diffPython(f1, f2):
    s1, s2 = os.stat(f1).st_size, os.stat(f2).st_size
    if s1 != s2:
        raise Exception("Files {} and {} do no have same size ({} != {})".format(f1, f2, s1, s2))

    blockSize = 4096
    with open(f1, "rb") as f:
        with open(f2, "rb") as g:
            while True:
                a = f.read(blockSize)
                b = g.read(blockSize)
                if a != b:
                    raise Exception("Read blocks differ in {} and {}".format(f1, f2))
                if len(a) == 0:
                    break

diff = diffPython

def main():
    if len(sys.argv) < 3:
        usage()

    firstExe = sys.argv[1]
    secondExe = sys.argv[2]
    args = sys.argv[3:]
    if len(args)%2 != 0:
        usage()

    optionAndValues = [ args[i:i+2] for i in range(0,len(args),2) ]

    inputFile = None
    outputBase = None
    numClusters = None
    repetitions = None
    seed = None
    traceBase = None
    centroidTraceBase = None
    blocks = None
    threads = None

    outputBases = [ ]

    for o,v in optionAndValues:
        if o == "--input":
            inputFile = v
        elif o == "--output":
            outputBase = v
            outputBases.append(outputBase)
        elif o == "--k":
            numClusters = int(v)
        elif o == "--repetitions":
            repetitions = int(v)
        elif o == "--seed":
            seed = int(v)
        elif o == "--trace":
            traceBase = v
            outputBases.append(traceBase)
        elif o == "--centroidtrace":
            centroidTraceBase = v
            outputBases.append(centroidTraceBase)
        elif o == "--blocks":
            blocks = int(v)
        elif o == "--threads":
            threads = int(v)
        else:
            usage()

    if inputFile is None or outputBase is None or numClusters is None or repetitions is None or seed is None:
        usage()

    suffixes = [ ".1", ".2" ]
    for exe, suffix in zip([firstExe, secondExe], suffixes):

        if exe == "SKIP":
            print("# Skipping reference run", file=sys.stderr)
            for f in [ out+suffix for out in outputBases ]:
                if not os.path.exists(f):
                    raise Exception("Output file {} does not exist, but is needed since we're skipping the reference run".format(f))
        else:
            cmd = [ exe, "--input", inputFile, "--k", str(numClusters),
                    "--output", outputBase+suffix, "--repetitions", str(repetitions) ]

            if seed is not None:
                cmd += [ "--seed", str(seed) ]
            if traceBase is not None:
                cmd += [ "--trace", traceBase+suffix ]
            if centroidTraceBase is not None:
                cmd += [ "--centroidtrace", centroidTraceBase+suffix ]
            if threads is not None:
                cmd += [ "--threads", str(threads) ]
            if blocks is not None:
                cmd += [ "--blocks", str(blocks) ]

            print("# Running", cmd, file=sys.stderr)
            subprocess.check_call(cmd)

        for f in [ out+suffix for out in outputBases ]:
            if not os.path.exists(f):
                raise Exception("Output file {} was not created".format(f))

    for out in outputBases:
        files = [ out+suffix for suffix in suffixes ]
        diff(*files)
        print("# Output checks out:", files, file=sys.stderr)

    print("# Done", file=sys.stderr)

if __name__ == "__main__":
    main()
