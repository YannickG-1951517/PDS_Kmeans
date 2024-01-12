#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:30:00
#SBATCH --error="n_proc_1_small-error"
#SBATCH --output="n_proc_1_small"

source /data/leuven/303/vsc30380/slurmhooks

module load iimpi/2022b

for i in {1..10}
do
    for j in {1..4}
    do
        mpirun -n $i ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads $i
    done
done