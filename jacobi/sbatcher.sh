#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="jacobi"
#SBATCH --get-user-env
#   #SBATCH --account=ict24_dssc_gpu
#  #SBATCH --partition=boost_usr_prod
#SBATCH --account=ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1                       # <--   Change there
#SBATCH --ntasks=1                      # <--   Change there
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
# #SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=02:00:00

nproc=1                                 # <--  Change there

# Personal remarsk:
# -  on boos_usr_prod:
#       1 socket --> 32 core, 1 socket per node
#       4 GPUs per node
# - on dcgp_usr_prot:
#       2 socket
#       56 core per socket 

matsize=12000
niter=10
dirout="./plots"
fileout="gpu-1200.csv"

# Requirements for the assignment
#   - 0. iteration: 10
#   - 1. matrix size: 1200 and 12000

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

# TODO: modify the Makefile according to the partition you are running the code
make

# TODO: set environment variables
#       according to your needs

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112

# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores
# export OMP_NUM_THREADS=56



#                                     <-- commented at the end to generate the reference pictures
# TODO: Comment after first time:
# echo "time,rank,size,what" > $dirout/$fileout
# mpirun -np $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

echo "time,rank,size,what" > multithreads_12k.csv
srun -N $nproc ./jacobi.x $matsize $niter >> multithreads_12k.csv

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
