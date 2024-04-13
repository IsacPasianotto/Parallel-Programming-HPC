#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="mmm-pt1"
#SBATCH --get-user-env
#SBATCH --account ict24_dssc_gpu
#  #SBATCH --partition=dcgp_usr_prod     # used to CPU computation
#SBATCH --partition=boost_usr_prod     # used to GPU computation
#SBATCH --nodes=64
#SBATCH --ntasks=256                   # Every MPI proc is a task --> = n nodes
#SBATCH --ntasks-per-node=4          # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:15:00


# on boos_usr_prod:
#  1socket 32core per node
#  4 GPUs per node

nproc=256      #number of MPI-processes
matsize=75000

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# TODO: change the module load commands
#       according to the architecture you are using

#  module load openMPI/4.1.5/gnu/
# module load openmpi/4.1.6--gcc--12.2.0
# module load openblas/0.3.24--gcc--12.2.0
module load openmpi/4.1.6--nvhpc--23.11
module load openblas/0.3.24--nvhpc--23.11


# Remove old files if any exist and then compile
make clean
make cuda

# TODO: set environment variables
#       according to your needs


#OMP_ variables are effective also for openBLAS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores
# export OMP_NUM_THREADS=56

# Comment after first time:
# echo "time(s),rank-worker,number-of-nodes,algorithm,thing-measured" > gpudata75k.csv

mpirun -np $nproc ./main.x $matsize >> gpudata75k.csv

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
