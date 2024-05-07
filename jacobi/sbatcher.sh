#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="jacobi"
#SBATCH --get-user-env
#SBATCH --account ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:15:00


# Personal remarsk:
# -  on boos_usr_prod:
#       1 socket --> 32 core, 1 socket per node
#       4 GPUs per node

nproc=8      #number of MPI-processes
matsize=75000
niter=1

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# TODO: change the module load commands
#       according to the architecture you are using

module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0

# Remove old files if any exist and then compile
make clean
make

# TODO: set environment variables
#       according to your needs

#OMP_ variables are effective also for openBLAS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores
# export OMP_NUM_THREADS=56

# Comment after first time:

mpirun -np $nproc ./main.x $matsize $niter

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
