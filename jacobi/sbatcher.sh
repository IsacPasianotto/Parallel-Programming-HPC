#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="jacobi"
#SBATCH --get-user-env
#SBATCH --account=ict24_dssc_gpu
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1                       # <--   Change there
#SBATCH --ntasks=4                      # <--   Change there
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=02:00:00


# Personal remarsk:
# -  on boos_usr_prod:
#       1 socket --> 32 core, 1 socket per node
#       4 GPUs per node
nproc=8      #number of MPI-processes

# Requirements for the assignment
#   - 0. iteration: 10
#   - 1. matrix size: 1200 and 12000

matsize=1200
niter=10
dirout="./plots"
fileout="cpu-1200.csv"

mkdir -p $dirout

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


# TODO: change the module load commands
#       according to the architecture you are using

module load openmpi/4.1.6--nvhpc--23.11

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
echo "time,rank,size,what" > $dirout/$fileout
mpirun -np $nproc ./jacobi.x $matsize $niter

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
