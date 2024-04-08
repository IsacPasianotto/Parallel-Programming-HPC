/*---------------------------------------------*
 | file: main.c                                |
 | author: Isac Pasianotto                     |
 | date: 2024-04                               |
 | context: exam of "Parallel programming for  |
 |          HPC". Msc Course in DSSC           |
 | description: main file for the exercise of  |
 |              matrix-matrix multiplication   |
 *---------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "init.h"
#include "debug.h"
#include "column_gathering.h"


int main(int argc, char* argv[])
{
  /*---------------------------------------------*
   | 0. Initialization of MPI environment         |
   *---------------------------------------------*/

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*--------------------------------------------------*
   | 1. Compute the local quantities for each worker  |
   |    and initialize the local matrices             |
   *--------------------------------------------------*/
  
  long int N = 2000;

#if defined(DEBUG) | defined(SMALL)
  N = 10;
#endif

  // TODO: if needed, evaluate an approach like:
  //    https://github.com/IsacPasianotto/foundations_of_HPC-assignment/blob/main/Assignment1/main.c
  if (argc > 1) {
    N = atoi(argv[1]);
  }

  long int local_size = (rank < N % size) ? N / size + 1 : N / size;
  
  // Allocate memory for the local matrices
  double* A = (double*) malloc(local_size * N * sizeof(double));
  double* B = (double*) malloc(local_size * N * sizeof(double));
  double* C = (double*) malloc(local_size * N * sizeof(double));

  init_local_matrix(A, local_size * N);
  init_local_matrix(B, local_size * N);

#if defined(DEBUG) | defined(DEBUG_INIT)
  debug_init_local_matrix(A, B, N, local_size, rank, size);
#endif

  // I will need a buffer to receive the blocks from the other processes
  long int buffer_size = ( (N % size) > 0 ) ? N/size + 1 : N/size;        // max possible size
  double* buffer = (double*) malloc(buffer_size * N * sizeof(double));

  /*--------------------------------------------------*
   | 2. Main loop over the number of processes to     |
   |    perform the local portion of the computation  |
   *--------------------------------------------------*/

  int* all_sizes = (int*) malloc(size * sizeof(int));

  for (int i = 0; i < size; i++)
  {
    all_sizes[i] = (i < N % size) ? N / size + 1 : N / size;
  }


  for (int iter = 0; iter < size; iter++)
  {
    /*--------------------------------------------------*
     | 2.1. Compute the block of the column             |
     *--------------------------------------------------*/

    double* local_block = (double*) malloc(local_size * all_sizes[iter] * sizeof(double));
    build_column_block(local_block, B, N, local_size, size, iter, all_sizes);

#if defined(DEBUG) | defined(DEBUG_COL_BLOCK)
    debug_col_block(B, local_block, N, local_size, rank, iter, all_sizes);
#endif




  } // loop over the number of processes



  /*--------------------------------------------------*
  | 3. Clean up everything                            |
  *--------------------------------------------------*/

  free(A);
  free(B);
  free(C);
  free(buffer);

  MPI_Finalize();
}
