#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>


#include <cuda_runtime.h>
#include <cublas.h>

#include "init.h"
#include "debug.h"
#include "column_gathering.h"
#include "product.h"
#include "stopwatch.h"

int main(int argc, char* argv[])
{
  /*---------------------------------------------*
   | 0. Initialization of MPI environment         |
   *---------------------------------------------*/

  // set number of threads equal to the number of cores in the current processor

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // see how many GPUs are available
  
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);
  printf("Number of GPUs: %d\n", nGPUs);
  cudaSetDevice(rank % nGPUs);
  printf("Rank %d is using GPU %d\n", rank, rank % nGPUs);
  
  MPI_Finalize();
  return 0;
}
