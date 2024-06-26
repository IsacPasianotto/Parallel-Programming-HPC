/*--------------------------------------------*
| file: product.c                             |
| author: Isac Pasianotto                     |
| date: 2024-04                               |
| context: exam of "Parallel programming for  |
|          HPC". Msc Course in DSSC           |
| description: functions that implements the  |
|      "naive" part of the exercise. It       |
|      performs a matrix-matrix multiplication|
|      with the classic 3-nested loop         |
*---------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

#include "product.h"

void compute_block_result_naive(double* local_C_block, double* A, double* buffer, long int N, long int local_size, int* all_sizes, int iter)
{
  // A: local_size x N
  // buffer: N x all_sizes[iter]
  // local_C_block: local_size x all_sizes[iter]
  #pragma omp parallel for collapse(2) 
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      double sum = 0.0;
      #pragma omp parallel for
      for (int k = 0; k < N; k++)
      {
        sum += A[i * N + k] * buffer[k * all_sizes[iter] + j];
      }
      local_C_block[i * all_sizes[iter] + j] = sum;
    } // loop over j
  } // loop over i
}

void copy_block_to_global_C(double* C, double* local_C_block, long int N, long int local_size, int* all_sizes, int size, int iter)
{
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      C[index + i * N + j] = local_C_block[i * all_sizes[iter] + j];
    }
  }
}