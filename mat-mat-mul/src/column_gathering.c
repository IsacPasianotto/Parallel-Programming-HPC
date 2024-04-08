#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "../include/column_gathering.h"

void build_column_block(double* local_block, double* B, long int N, long int local_size, int size, int iter, int* all_sizes)
{
  // index used to extract the block from the matrix B
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      local_block[i * all_sizes[iter] + j] = B[index];
      index++;
    }
    index = iter * ((N % size) > 0 ? N / size + 1 : N / size) + (i + 1) * N;  // go to the next column
  }

}