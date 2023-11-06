#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=llp_h_pds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

for i in {1..4}
do
echo "omp kmeans $i"
./omp/kmeans --input ./datasets/1M_1000000x4.csv --output kmeans_test_own_code --k 7 --repetitions 15 --seed 1951517 --threads 1
done

for i in {1..4}
do
echo "serial kmeans $i"
./serial/kmeans --input ./datasets/1M_1000000x4.csv --output kmeans_test_own_code --k 7 --repetitions 15 --seed 1951517
done