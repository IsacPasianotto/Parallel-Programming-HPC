#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="jacobi"
#SBATCH --get-user-env
#SBATCH --account=ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=16                       # <--   Change there
#SBATCH --ntasks=16                      # <--   Change there
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=480G
#SBATCH --time=00:10:00

nproc=16                                 # <--  Change there
nnodes="16nodes"                         # <--  Change there

# Requirements for the assignment
#   - 0. iteration: 10
#   - 1. matrix size: 1200 and 12000

dirout="./plots"
export niter=10
mkdir -p $dirout

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


module load openmpi/4.1.6--gcc--12.2.0

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=112
# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores

############################
##  RMA:  One win - PUT  ##
############################
make clean
make

nwin="1win"
type="put"
matsize=1200

fileout="$nnodes-$nwin-$type-$matsize.csv"


make clean
make

echo "time,rank,size,what" > $dirout/$fileout
for i in {1..10};do
    srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout
done

echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
