#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="mmm-pt1"
#SBATCH --get-user-env
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod     # used to CPU computation
#  #SBATCH --partition=boost_usr_prod     # used to GPU computation
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # Every MPI proc is a task --> = n nodes
#SBATCH --ntasks-per-node=1          # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=112            # Number of CPU cores per task
#  #SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:15:00


# on boos_usr_prod:
#  1socket 32core per node
#  4 GPUs per node

nproc=1      #number of MPI-processes
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

#  module load openMPI/4.1.5/gnu/
module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
# module load openmpi/4.1.6--nvhpc--23.11
# module load openblas/0.3.24--nvhpc--23.11


# Remove old files if any exist and then compile
make clean
make blas

# TODO: set environment variables
#       according to your needs


#OMP_ variables are effective also for openBLAS
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores
# export OMP_NUM_THREADS=56

filename="multithreads.csv"
# echo "time(s),rank-worker,number-of-nodes,algorithm,thing-measured" > $filename

srun -N $nproc ./main.x $matsize >> $filename

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
