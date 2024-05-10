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

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef CUDA
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
#ifndef CUDA
  double* time_records = (double*) malloc( (2 + 5*size) * sizeof(double));
#else 
  double* time_records = (double*) malloc( (3 + 9*size)* sizeof(double));
#endif

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
  double *d_A;
  cudaMalloc((void **) &d_A, local_size * N * sizeof(double));
  cudaMemcpy(d_A, A, local_size * N * sizeof(double), cudaMemcpyHostToDevice);
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
    record_time(time_records, time_counter);  //t_{2+ 5 * iter}  ;  t_cuda_{3 + 9 * iter}

    long int buffer_size =  (iter < N % size) ? N / size + 1 : N / size;
    double* buffer = (double*) malloc(buffer_size * N * sizeof(double));
    memset(buffer, 0.0, buffer_size * N * sizeof(double));

    double* local_block = (double*) malloc(local_size * all_sizes[iter] * sizeof(double));
    build_column_block(local_block, B, N, local_size, size, iter, all_sizes);

    record_time(time_records, time_counter);  //t_{3+ 5 * iter} ;  t_cuda_{4 + 9 * iter}

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

    record_time(time_records, time_counter);  //t_{4+ 5 * iter}  ;  t_cuda_{5 + 9 * iter}

    MPI_Allgatherv(local_block, local_size * all_sizes[iter], MPI_DOUBLE, buffer, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{6 + 9 * iter}

#if defined(DEBUG) | defined(DEBUG_COL_GATHER)
    debug_allgatherv(B, local_block, buffer, N, local_size, rank, iter, all_sizes, buffer_size);
#endif

    /*--------------------------------------------------*
     | 2.3. Compute the local portion of the product    |
     *--------------------------------------------------*/

    double* local_C_block = local_block;    // local_block is not needed anymore, so we can reuse it and save memory
    memset(local_C_block, 0.0, local_size * all_sizes[iter] * sizeof(double));

#ifdef CUDA
    
    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{7 + 9 * iter}
    
    // move data to GPU
    double* device_C_block;
    double* device_B_buffer;
    cudaMalloc((void **) &device_C_block, local_size * all_sizes[iter] * sizeof(double));
    cudaMalloc((void **) &device_B_buffer, buffer_size * N * sizeof(double));
    cudaMemcpy(device_B_buffer, buffer, buffer_size * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(device_C_block, 0.0, local_size * all_sizes[iter] * sizeof(double));

    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{8 + 9 * iter}

    // Perform the computation
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, all_sizes[iter], local_size, N, &alpha, device_B_buffer, all_sizes[iter], d_A, N, &beta, device_C_block, all_sizes[iter]);
    cublasDestroy(handle);

    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{9 + 9 * iter}

    // copy the result back to the host
    cudaMemcpy(local_C_block, device_C_block, local_size * all_sizes[iter] * sizeof(double), cudaMemcpyDeviceToHost);

    record_time(time_records, time_counter);  //t_{5+ 5 * iter}  ;  t_cuda_{10 + 9 * iter}

#else // not CUDA
#ifdef OPENBLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, local_size, all_sizes[iter], N, 1.0, A, N, buffer, all_sizes[iter], 1.0, local_C_block, all_sizes[iter]);
#else
    compute_block_result_naive(local_C_block, A, buffer, N, local_size, all_sizes, iter);
#endif // ifdef OPENBLAS
#endif // ifdef CUDA

    copy_block_to_global_C(C, local_C_block, N, local_size, all_sizes, size, iter);
    record_time(time_records, time_counter);  //t_{6+ 5 * iter} ;  t_cuda_{11 + 9 * iter}

    /*--------------------------------------------------*
    | 2.4. Clean up memory for the loop                |
    *--------------------------------------------------*/

#ifdef CUDA
    cudaFree(device_C_block);
    cudaFree(device_B_buffer);
#endif
    free(sendcounts);
    free(displs);
    free(buffer);
    free(local_block);

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
   | 4. Clean up everything                           |
   *--------------------------------------------------*/

  free(A);
  free(B);
  free(C);
  free(time_records);
  free(all_sizes);


#ifdef CUDA
  cudaFree(d_A);
#endif

  MPI_Finalize();

  return 0;
}
