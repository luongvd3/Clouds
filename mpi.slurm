#!/bin/bash
#SBATCH --error=err-%j.err
#SBATCH --output=snowy-1-4-%j.out
#SBATCH --partition=snowy
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --job-name=pure_mpi


module load openmpi/3.1.4
module load mpi4py/3.0.2-timed-pingpong


echo "Scheduler 1 nodes 4 task 4 cpus (snowy)"
mpiexec -n 4 python MpiMain.py
