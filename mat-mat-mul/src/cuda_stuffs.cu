#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>


#include <cuda_runtime.h>
#include <cublas_v2.h>
#define N_THREADS_PER_BLOCK 32

#include "../include/init.h"
#include "../include/debug.h"
#include "../include/column_gathering.h"
#include "../include/product.h"
#include "../include/stopwatch.h"



void assign_gpu_to_process(int rank)
{
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  cudaSetDevice(rank % n_gpus);
}

void get_ready_on_gpu(double* A, double* C, double* d_A, double* d_C, long int N, long int local_size, int rank, int size, double* time_records, int* time_counter)
{
  cudaMalloc((void **) &d_A, local_size * N * sizeof(double));
  cudaMalloc((void **) &d_C, local_size * N * sizeof(double));
  cudaMemcpy(d_A, A, local_size * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, local_size * N * sizeof(double), cudaMemcpyHostToDevice);

  record_time(time_records, time_counter);  // -- , t_cuda_2
}

__global__ void cuda_copy_block_to_global_c(double* d_C, double* local_C_block, long int N, long int local_size, int* all_sizes, int size, int iter)
{
  long int index = iter * ((N % size) > 0 ? N / size + 1 : N / size);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < local_size && j < all_sizes[iter])
  {
    d_C[index + i * N + j] = local_C_block[i * all_sizes[iter] + j];
  }
}

void compute_block_result_cuda(double* d_A, double* d_C, double* buffer, long int N, long int local_size, int* all_sizes, int size, int iter, double* time_records, int* time_counter)
{

  record_time(time_records, time_counter);  // --- ;  t_cuda_{7 + 7 * iter}

  double *device_C_block;
  double *device_B_buffer;
  cudaMalloc((void **) &device_C_block, local_size * all_sizes[iter] * sizeof(double));
  cudaMalloc((void **) &device_B_buffer, buffer_size * N * sizeof(double));

  cudaMemcpy(device_B_buffer, buffer, buffer_size * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(device_C_block, 0.0, local_size * all_sizes[iter] * sizeof(double));

  record_time(time_records, time_counter);  // --- ;  t_cuda_{8 + 7 * iter}

  // perform the product
  cublasHandle_t handle;
  cublasCreate(&handle);
  double alpha = 1.0;
  double beta = 1.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, local_size, all_sizes[iter], N, &alpha, d_A, N, device_B_buffer, all_sizes[iter], &beta, device_C_block, all_sizes[iter]);

  dim3 threads(N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK);
  dim3 blocks((local_size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, (all_sizes[iter] + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK);
  cuda_copy_block_to_global_c<<<blocks, threads>>>(d_C, device_C_block, N, local_size, all_sizes, size, iter);

  cudaDeviceSynchronize();  // wait for the kernel to finish
  record_time(time_records, time_counter);  // --- ;  t_cuda_{9 + 7 * iter}
}


void free_gpu_memory_loop(double* device_C_block, double* device_B_buffer, cublasHandle_t handle)
{
  cublasDestroy(handle);
  cudaFree(device_C_block);
  cudaFree(device_B_buffer);
}

void free_gpu_memory(double* d_A, double* d_C)
{
  cudaFree(d_A);
  cudaFree(d_C);
}
