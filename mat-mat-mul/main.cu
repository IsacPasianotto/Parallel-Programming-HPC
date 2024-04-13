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
#include <string.h>
#include <omp.h>
#include <mpi.h>

#ifdef OPENBLAS
#include <cblas.h>
#endif

#ifdef CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define N_THREADS_PER_BLOCK 32
#endif

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

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef CUDA
  // each process will use its own GPU
  int n_gpus;
  cudaGetDeviceCount(&n_gpus);
  cudaSetDevice(rank % n_gpus);
#endif


  /*--------------------------------------------------*
   | 1. Compute the local quantities for each worker  |
   |    and initialize the local matrices             |
   *--------------------------------------------------*/

  long int N = 2000; // default value to override with the command line argument

#if defined(DEBUG) | defined(SMALL)
  N = 10;
#endif

  if (argc > 1) {
    N = atoi(argv[1]);
  }

  int tc = 0;
  int* time_counter = &tc;
  double* time_records = (double*) malloc( (2 + 5*size) * sizeof(double));
  memset(time_records, 0.0, (2 + 5*size) * sizeof(double));

  record_time(time_records, time_counter);    //t0 ; t_cuda_0

  long int local_size = (rank < N % size) ? N / size + 1 : N / size;

  // Allocate memory for the local matrices
  double* A = (double*) malloc(local_size * N * sizeof(double));
  double* B = (double*) malloc(local_size * N * sizeof(double));
  double* C = (double*) malloc(local_size * N * sizeof(double));

  init_local_matrix(A, local_size * N);
  init_local_matrix(B, local_size * N);
  memset(C, 0.0, local_size * N * sizeof(double));

  record_time(time_records, time_counter);  //t1 ; t_cuda_1

#ifdef CUDA
  // need pointers to the device memory
  double* d_A;
  double* d_B;
  double* d_C;
  cudaMalloc((void**)&d_A, local_size * N * sizeof(double));
  cudaMalloc((void**)&d_C, local_size * N * sizeof(double));
  cudaMemcpy(d_A, A, local_size * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, local_size * N * sizeof(double), cudaMemcpyHostToDevice);

  record_time(time_records, time_counter);  // -- , t_cuda_2

#endif

#if defined(DEBUG) | defined(DEBUG_INIT)
  debug_init_local_matrix(A, B, N, local_size, rank, size);
#endif

  int* all_sizes = (int*) malloc(size * sizeof(int));

  for (int i = 0; i < size; i++)
  {
    all_sizes[i] = (i < N % size) ? N / size + 1 : N / size;
  }

  /*--------------------------------------------------*
  | 2. Main loop over the number of processes to     |
  |    perform the local portion of the computation  |
  *--------------------------------------------------*/

  for (int iter = 0; iter < size; iter++)
  {
    /*--------------------------------------------------*
     | 2.1. Compute the block of the column             |
     *--------------------------------------------------*/

    record_time(time_records, time_counter);  //t_{2+ 5 * iter}  ;  t_cuda_{3 + 7 * iter}

    long int buffer_size =  (iter < N % size) ? N / size + 1 : N / size;
    double* buffer = (double*) malloc(buffer_size * N * sizeof(double));
    memset(buffer, 0.0, buffer_size * N * sizeof(double));

    double* local_block = (double*) malloc(local_size * all_sizes[iter] * sizeof(double));
    build_column_block(local_block, B, N, local_size, size, iter, all_sizes);

    record_time(time_records, time_counter);  //t_{3+ 5 * iter} ;  t_cuda_{4 + 7 * iter}

#if defined(DEBUG) | defined(DEBUG_COL_BLOCK)
    debug_col_block(B, local_block, N, local_size, rank, iter, all_sizes);
#endif

    /*--------------------------------------------------*
     | 2.2. Do an ALLGATHER operation to collect data   |
     *--------------------------------------------------*/

    int* sendcounts = (int*) malloc(size * sizeof(int));
    int* displs = (int*) malloc(size * sizeof(int));     //displacements

    compute_receive_counts(sendcounts, all_sizes, size, iter);
    compute_displacements(displs, sendcounts, size);

    record_time(time_records, time_counter);  //t_{4+ 5 * iter}  ;  t_cuda_{5 + 7 * iter}

    MPI_Allgatherv(local_block, local_size * all_sizes[iter], MPI_DOUBLE, buffer, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{6 + 7 * iter}

#if defined(DEBUG) | defined(DEBUG_COL_GATHER)
    debug_allgatherv(B, local_block, buffer, N, local_size, rank, iter, all_sizes, buffer_size);
#endif

    /*--------------------------------------------------*
     | 2.3. Compute the local portion of the product    |
     *--------------------------------------------------*/


#ifdef CUDA

    record_time(time_records, time_counter);  // --- ;  t_cuda_{7 + 7 * iter}

    double* device_C_block;
    double* device_B_buffer;
    cudaMalloc((void**)&device_C_block, local_size * all_sizes[iter] * sizeof(double));
    cudaMalloc((void**)&device_B_buffer, buffer_size * N * sizeof(double));

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
    // cuda_copy_block_to_global_c<<<blocks, threads>>>(d_C, device_C_block, N, local_size, all_sizes, size, iter);
    cuda_copy_block_to_global_c<<<blocks, threads>>>(d_C, device_C_block, N, local_size, all_sizes, size, iter);


    cudaDeviceSynchronize();  // wait for the kernel to finish
    record_time(time_records, time_counter);  // --- ;  t_cuda_{9 + 7 * iter}

#else // not CUDA

    // double* local_C_block = (double*) malloc(local_size * all_sizes[iter] * sizeof(double));
    // local_block is not needed anymore, so we can reuse it and save memory
    double* local_C_block = local_block;
    memset(local_C_block, 0.0, local_size * all_sizes[iter] * sizeof(double));

#ifdef OPENBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, local_size, all_sizes[iter], N, 1.0, A, N, buffer, all_sizes[iter], 1.0, local_C_block, all_sizes[iter]);
#else
    compute_block_result_naive(local_C_block, A, buffer, N, local_size, all_sizes, iter);
#endif
    copy_block_to_global_C(C, local_C_block, N, local_size, all_sizes, size, iter);
    record_time(time_records, time_counter);  //t_{6+ 5 * iter} ;  ---
#endif // ifdef CUDA

    /*--------------------------------------------------*
     | 2.4. Clean up memory for the loop                |
     *--------------------------------------------------*/

    free(sendcounts);
    free(displs);
    free(buffer);

#ifdef CUDA
    cublasDestroy(handle);
    cudaFree(device_C_block);
    cudaFree(device_B_buffer);
#endif

#ifndef CUDA
    free(local_C_block);
#endif

  } // loop over the number of processes

  /*--------------------------------------------------*
   | 3. Loop is over: print the results!              |
   *--------------------------------------------------*/

#if defined(DEBUG) | defined(DEBUG_PROD)
  debug_product(A, B, C, N, local_size, rank, size);
#endif

#ifdef CUDA
  const char* program_type = "GPU-Cuda";
#endif
#ifdef OPENBLAS
  const char* program_type = "CPU-OpenBLAS";
#endif
#if !defined(OPENBLAS) && !defined(CUDA)
  const char* program_type = "CPU-naive";
#endif

#ifndef NOSTOPWATCH
#ifndef CUDA
  print_time_records(time_records, rank, size, program_type);
#else
  print_time_records_cuda(time_records, rank, size, program_type);
#endif
#endif

  /*--------------------------------------------------*
  | 4. Clean up everything                            |
  *--------------------------------------------------*/

  free(A);
  free(B);
  free(C);
  free(all_sizes);
  free(time_records);

#ifdef CUDA
  cudaFree(d_A);
  cudaFree(d_C);
#endif

  MPI_Finalize();

  return 0;
}
