#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="mmm-pt1"
#SBATCH --get-user-env
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16
#SBATCH --ntasks=16                   # Every MPI proc is a task --> = n nodes
#SBATCH --ntasks-per-node=1          # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=56           # Number of CPU cores per task
#SBATCH --mem=15G
#SBATCH --time=02:00:00

nproc=16      #number of MPI-processes
matsize=5000

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# TODO: change the module load commands
#       according to the architecture you are using
module purge
#  module load openMPI/4.1.5/gnu/
module load openmpi/4.1.6--gcc--12.2.0

# Remove old files if any exist and then compile
make clean
make

# TODO: set environment variables
#       according to your needs


export OMP_PLACES=cores
export OMP_PROC_BIND=close

# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores
# export OMP_NUM_THREADS=56

# Comment after first time:
# echo "time(s),rank-worker,number-of-nodes,algorithm,thing-measured" > data.csv

mpirun -np $nproc ./main.x $matsize >> data.csv

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
