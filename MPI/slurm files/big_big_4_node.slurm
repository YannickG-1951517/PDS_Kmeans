#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:30:00
#SBATCH --error="big_big_4-error"
#SBATCH --output="big_big_4"

source /data/leuven/303/vsc30380/slurmhooks

module load iimpi/2022b

for i in {1..4}
do
    mpirun -n 128 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 128
done


for i in {1..4}
do
    mpirun -n 256 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 256
done


for i in {1..4}
do
    mpirun -n 288 ./kmeans_mpi --input ../datasets/1M_1000000x4.csv --output outputfile.csv --k 7 --repetitions 15 --seed 1951517 --threads 288
done
