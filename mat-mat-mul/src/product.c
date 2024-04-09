#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "product.h"

void compute_block_result_naive (double* local_C_block, double* A, double* buffer, long int N, long int local_size, int* all_sizes, int iter)
{
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      for (int k = 0; k < N; k++)
      {
        local_C_block[i * all_sizes[iter] + j] += A[i * N + k] * buffer[k * all_sizes[iter] + j];
      }
    }
  }
}

void copy_block_to_global_C (double* C, double* local_C_block, long int N, long int local_size, int* all_sizes, int size, int iter)
{
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      C[index] = local_C_block[i * all_sizes[iter] + j];
      index++;
    }
    index = iter * ((N % size) > 0 ? N / size + 1 : N / size) + (i + 1) * N;  // go to the next column
  }
}