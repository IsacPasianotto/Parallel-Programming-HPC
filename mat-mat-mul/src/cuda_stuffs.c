#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "../include/init.h"
#include "../include/debug.h"
#include "../include/column_gathering.h"
#include "../include/product.h"
#include "../include/stopwatch.h"
#include "../include/cuda_stuffs.h"


#ifdef CUDA
void assign_gpu_to_process(int rank)
{
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  cudaSetDevice(rank % n_gpus);
}

void get_ready_on_gpu(double* A, double* d_A, long int N, long int local_size, int rank, int size, double* time_records, int* time_counter)
{
  cudaMalloc((void **) &d_A, local_size * N * sizeof(double));
  cudaMemcpy(d_A, A, local_size * N * sizeof(double), cudaMemcpyHostToDevice);
  record_time(time_records, time_counter);  // -- , t_cuda_2
}

void compute_block_result_cuda(double* d_A, double* buffer, double* local_C_block, double*device_C_block, double* device_B_buffer, long int buffer_size, long int N, long int local_size, int* all_sizes, int size, int iter, double* time_records, int* time_counter)
{

  record_time(time_records, time_counter);  // --- ;  t_cuda_{7 + 9 * iter}

  cudaMalloc((void **) &device_C_block, local_size * all_sizes[iter] * sizeof(double));
  cudaMalloc((void **) &device_B_buffer, buffer_size * N * sizeof(double));

  cudaMemcpy(device_B_buffer, buffer, buffer_size * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(device_C_block, 0.0, local_size * all_sizes[iter] * sizeof(double));

  record_time(time_records, time_counter);  // --- ;  t_cuda_{8 + 9 * iter}

  // perform the product
  cublasHandle_t handle;
  cublasCreate(&handle);
  double alpha = 1.0;
  double beta = 1.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_size, all_sizes[iter], N, &alpha, d_A, N, device_B_buffer, all_sizes[iter], &beta, device_C_block, all_sizes[iter]);
  cublasDestroy(handle);
  record_time(time_records, time_counter);  // --- ;  t_cuda_{9 + 9 * iter}

  // pass the result to host
  cudaMemcpy(local_C_block, device_C_block, local_size * all_sizes[iter] * sizeof(double), cudaMemcpyDeviceToHost);
  record_time(time_records, time_counter);  // --- ;  t_cuda_{10 + 9 * iter}

}


void free_gpu_memory_loop(double* device_C_block, double* device_B_buffer)
{
  cudaFree(device_C_block);
  cudaFree(device_B_buffer);
}

#endif
