#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

./serial/serial_kmeans --input ./datasets/1M_1000000x4.csv --output kmeans_test_own_code --k 7 --repetitions 15 --seed 1951517
./serial/omp_kmeans --input ./datasets/1M_1000000x4.csv --output kmeans_test_own_code --k 7 --repetitions 15 --seed 1951517

