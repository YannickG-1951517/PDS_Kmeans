#!/bin/bash

usage() {
	cat << EOF

Usage:

  ./mpiwrapper.sh ... --threads N ...

Set the EXECUTABLE environment variable to specify the actual command.

This is a helper script to use MPI programs together with the 'compare.py'
script: 'compare.py' expects a single command name, where the --threads option
can specify a number of threads. This 'mpiwrapper.sh' script detects that
option and starts 'mpirun' with the specified number of nodes. The actual
executable to run must be specified using the EXECUTABLE environment variable.

Running

  EXECUTABLE=./kmeans_mpi ./mpiwrapper.sh --input in.csv --output out.csv --k 3 --repetitions 4 --threads 2

will be transformed into

  mpirun -n 2 ./kmeans_mpi --input in.csv --output out.csv --k 3 --repetitions 4 --threads 2

Together with the compare.py script you could do something like

  EXECUTABLE=./kmeans_mpi ./compare.py ./kmeans_serial ./mpiwrapper.sh --input in.csv --output out.csv --k 3 --repetitions 4 --threads 2 --seed 12345

EOF
	exit -1
}

if [ "$#" -lt 2 ] ; then
	usage
fi

EXE="$EXECUTABLE"
if [ -z "$EXE" ] ; then
	usage
fi

NUMNODES=1
NEWARGS=""
while [ "$#" != 0 ] ; do
	OPT="$1"
	VAL="$2"
	NEWARGS="$NEWARGS $OPT $VAL"
	shift 2

	if [ "$OPT" == "--threads" ] ; then
		NUMNODES="$VAL"
	fi
done

mpirun -n $NUMNODES $EXE $NEWARGS

