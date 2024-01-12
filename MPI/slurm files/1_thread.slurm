#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --error="1_thread-error"
#SBATCH --output="1_thread"

source /data/leuven/303/vsc30380/slurmhooks

module load iimpi/2022b

for i in {1..4}
do
    mpirun -n 1 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 1
done
