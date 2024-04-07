#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="mmm"
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=2
#SBATCH --time=02:00:00

np=2      #number of MPI-processes

# To know each slurm.out file belongs to which job
date
hostname
whoami
pwd


# TODO: change the module load commands
#       according to the architecture you are using
module purge
module load openMPI/4.1.5/gnu/


# Remove old files if any exist and then compile
make clean
make

# TODO: set environment variables
#       according to your needs

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8

mpirun -np $np ./main.x
