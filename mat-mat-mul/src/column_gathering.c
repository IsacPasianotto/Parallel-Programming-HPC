#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "../include/column_gathering.h"

void build_column_block(double* local_block, double* B, long int N, long int local_size, int size, int iter, int* all_sizes)
{
  // index used to extract the block from the matrix B
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  #pragma omp parallel for
  for (int i = 0; i < local_size; i++)
  {
    #pragma omp parallel for
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      local_block[i * all_sizes[iter] + j] = B[index];
      index++;
    }
    index = iter * ((N % size) > 0 ? N / size + 1 : N / size) + (i + 1) * N;  // go to the next column
  }

}


void compute_receive_counts(int* revcounts, int* all_sizes, int size, int iter)
{
  #pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    revcounts[i] = all_sizes[iter] * all_sizes[i];
  }
}

void compute_displacements(int* displs, int* revcounts, int size)
{
  #pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    //displs[i] = (i == 0) ? 0 : displs[i-1] + revcounts[i-1];
    // recursion is hard to parallelize
    displs[i] = 0;
    if (i > 0)
    {
      #pragma omp parallel for
      for (int j = 0; j < i; j++)
      {
        displs[i] += revcounts[j];
      }
    }
  }
}