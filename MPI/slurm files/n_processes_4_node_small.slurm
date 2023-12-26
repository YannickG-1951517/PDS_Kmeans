#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=72
#SBATCH --time=02:00:00
#SBATCH --error="n_proc_4_small-error"
#SBATCH --output="n_proc_4_small"

source /data/leuven/303/vsc30380/slurmhooks

for i in {1..16}
do
    for j in {1..4}
    do
        EXECUTABLE=./kmeans_mpi ./mpiwrapper.sh --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads $i
    done
done