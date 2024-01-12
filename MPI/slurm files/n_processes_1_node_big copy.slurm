#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:30:00
#SBATCH --error="n_proc_1_big-error"
#SBATCH --output="n_proc_1_big"

source /data/leuven/303/vsc30380/slurmhooks

module load iimpi/2022b

for i in {1..4}
do
    mpirun -n 32 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 32
done


for i in {1..4}
do
    mpirun -n 64 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 64
done


for i in {1..4}
do
    mpirun -n 72 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 72
done
