#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#include "product.h"

void compute_block_result_naive(double* local_C_block, double* A, double* buffer, long int N, long int local_size, int* all_sizes, int iter) {
  // A: local_size x N
  // buffer: N x all_sizes[iter]
  // local_C_block: local_size x all_sizes[iter]
  for (int i = 0; i < local_size; i++) {
    for (int j = 0; j < all_sizes[iter]; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * buffer[k * all_sizes[iter] + j];
      }
      local_C_block[i * all_sizes[iter] + j] = sum;
    }
  }
}

void copy_block_to_global_C(double* C, double* local_C_block, long int N, long int local_size, int* all_sizes, int size, int iter) {
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  for (int i = 0; i < local_size; i++) {
    for (int j = 0; j < all_sizes[iter]; j++) {
      C[index + i * N + j] = local_C_block[i * all_sizes[iter] + j];
    }
  }
}
