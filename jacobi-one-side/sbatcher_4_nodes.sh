#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="jacobi"
#SBATCH --get-user-env
#SBATCH --account=ict24_dssc_cpu
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=4                      # <--   Change there
#SBATCH --ntasks=4                      # <--   Change there
#SBATCH --ntasks-per-node=1
#SBATHC --exclusive
#SBATCH --mem=480G
#SBATCH --time=02:00:00

nproc=4                                 # <--  Change there
nnodes="4nodes"
# Personal remarsk:
# - on dcgp_usr_prot:
#       2 socket
#       56 core per socket


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
# if OMP_NUM_THREADS is not set, the number of threads is equal to the number of cores

############################
##  RMA:  One win - PUT  ##
############################
make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -DONESIDE -DONEWIN -DPUT -o jacobi.x

nwin="1win"
type="put"
matsize=1200
fileout="$nnodes-$nwin-$type-$matsize.csv"


make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -DONESIDE -DONEWIN -DPUT -o jacobi.x

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

nwin="1win"
type="put"
matsize=12000
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout


############################
##  RMA:  One win - GET  ##
############################
make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -DONESIDE -DONEWIN -o jacobi.x

nwin="1win"
type="get"
matsize=1200
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

nwin="1win"
type="get"
matsize=12000
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

############################
##  RMA:  Two win - PUT  ##
############################
make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -DONESIDE -DPUT -o jacobi.x

nwin="2win"
type="put"
matsize=1200
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

nwin="2win"
type="put"
matsize=12000
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

############################
##  RMA:  Two win - GET  ##
############################
make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -DONESIDE -o jacobi.x

nwin="2win"
type="get"
matsize=1200
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

nwin="2win"
type="get"
matsize=12000
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

###########################
##       No RMA          ##
###########################
make clean
mpicc jacobi.c -O3 -Wall -fopenmp -DSTOPWATCH -o jacobi.x

nwin="two-sided-comm"
type="none"
matsize=1200
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout

nwin="two-sided-comm"
type="none"
matsize=12000
fileout="$nnodes-$nwin-$type-$matsize.csv"

echo "time,rank,size,what" > $dirout/$fileout
srun -N $nproc ./jacobi.x $matsize $niter >> $dirout/$fileout


echo "........................."
echo "   DONE!"
echo "  ended at $(date)"
echo "..........................."
