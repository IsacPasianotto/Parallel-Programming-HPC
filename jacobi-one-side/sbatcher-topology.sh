#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="topology"
#SBATCH --get-user-env
#SBATCH --account ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod         # CPU partition on LEONARDO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # Number of tasks (or processes) per node
#SBATCH --exclusive                       # Request the whole node to complete mapping
#SBATCH --time=00:05:00


# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "hostname:            $(hostname)"
echo "DATE:                $(date)"
echo "---------------------------------------------"


module load openmpi/4.1.6--nvhpc--23.11

# Remove old files if any exist and then compile
make clean

export workdir=$(pwd)
export LD_LIBRARY_PATH=$workdir/hwloc-build/lib:$LD_LIBRARY_PATH

mpicc topology-explorer.c -O3 -I$workdir/hwloc-build/include -L$workdir/hwloc-build/lib -lhwloc -o topology-explorer.x
mpirun -np 1 ./topology-explorer.x


echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
