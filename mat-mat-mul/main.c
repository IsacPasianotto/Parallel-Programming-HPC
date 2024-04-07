/*---------------------------------------------*
 | file: main.c                                |
 | author: Isac Pasianotto                     |
 | date: 2024-04                               |
 | context: exam of "Parallel programming for  |
 |          HPC". Msc Course in DSSC           |
 | description: main file for the exercise of  |
 |              matrix matrix multiplication   |
 *---------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "init.h"


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
  
  long int N = 20; // small default value 
  // TODO: if needed, evaluate an approach like:
  //    https://github.com/IsacPasianotto/foundations_of_HPC-assignment/blob/main/Assignment1/main.c
  if (argc > 1) {
    N = atoi(argv[1]);
  }

  long int local_size = (rank < N % size) ? N / size + 1 : N / size;
  
  // Allocate memory for the local matrices
  double* A = (double*) malloc(local_size * N * sizeof(double));
  double* B = (double*) malloc(local_size * N * sizeof(double));

  init_local_matrix(A, local_size * N);
  init_local_matrix(B, local_size * N);

  
#ifdef DEBUG
  if (rank == 0)
    printf("local_size = %ld\n", local_size);
  printf("Rank: %d:\t A[0]: %f\t B[0]: %f\n", rank, A[0], B[0]);
  printf("Rank: %d:\t A[%ld]: %f\t B[%ld]: %f\n", rank, local_size * N - 1, A[local_size * N - 1], local_size * N - 1, B[local_size * N - 1]);
#endif


  MPI_Finalize();
}
