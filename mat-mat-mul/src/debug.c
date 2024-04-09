#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#include "debug.h"

void debug_init_local_matrix(double* A, double* B, long int N, long int local_size, int rank, int size)
{
  if (rank == 0)
  {
    printf("---------debug init---------\n");
    printf("N: %ld\n", N);
    printf("Total entries in A: %ld\n", N * N);
    printf("Number of processes: %d\n", size);
    printf("local_size (process 0): %ld\n", local_size);
    // some print to check the initialization gone well
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++)
  {
    if (rank == i)
    {
      printf("Rank: %d:\t A[0]: %f\t B[0]: %f\n", rank, A[0], B[0]);
      printf("Rank: %d:\t A[%ld]: %f\t B[%ld]: %f\n", rank, local_size * N - 1, A[local_size * N - 1], local_size * N - 1, B[local_size * N - 1]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("---------debug init---------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stdout);
}

void debug_col_block(double* B, double* local_block, long int N, long int local_size, int rank, int iter, int* all_sizes)
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("---------debug column block---------\n");
    printf("LOCAL MATRIX B:\n");
    for (int i = 0; i < local_size; i++)
    {
      for (int j = 0; j < N; j++)
      {
        printf("%f ", B[i * N + j]);
      }
      printf("\n");
    }
    printf("LOCAL BLOCK:\n");
    for (int i = 0; i < local_size; i++)
    {
      for (int j = 0; j < all_sizes[iter]; j++)
      {
        printf("%f ", local_block[i * all_sizes[iter] + j]);
      }
      printf("\n");
    }
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  printf("---------debug column block---------\n");
  MPI_Barrier(MPI_COMM_WORLD);
}

void debug_allgatherv(double* B, double* local_block, double* buffer, long int N, long int local_size, int rank, int iter, int* all_sizes, int buffer_size)
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("---------debug allgatherv---------\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("I'm rank %d\titer %d\n", rank, iter);
  printf("My part of B is:\n");
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf("%f ", B[i * N + j]);
    }
    printf("\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("..................\n");
  printf("I'm rank %d\titer %d\n", rank, iter);
  printf("My local block is:\n");
  for (int i = 0; i < local_size; i++)
  {
    for (int j = 0; j < all_sizes[iter]; j++)
    {
      printf("%f ", local_block[i * all_sizes[iter] + j]);
    }
    printf("\n");
  }
  printf("..................\n");
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("Received buffer from all processes:\n");
    for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < buffer_size; j++)
      {
        printf("%f ", buffer[i * buffer_size + j]);
      }
      printf("\n");
    }
    printf("---------debug allgatherv---------\n");
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
}


void debug_product(double* A, double* B, double* C, long int N, long int local_size, int rank, int size)
{
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++)
  {
    if (rank == i)
    {
      printf("............................\n");
      printf("I'm rank %d\n", rank);
      printf("My copy of A is: \n");
      for (int i = 0; i < local_size; i++)
      {
        for (int j = 0; j < N; j++)
        {
          printf("%f ", A[i * N + j]);
        }
        printf("\n");
      }
      printf("my copy of B is: \n");
      for (int i = 0; i < local_size; i++)
      {
        for (int j = 0; j < N; j++)
        {
          printf("%f ", B[i * N + j]);
        }
        printf("\n");
      }
      printf("my copy of C is: \n");
      for (int i = 0; i < local_size; i++)
      {
        for (int j = 0; j < N; j++)
        {
          printf("%f ", C[i * N + j]);
        }
        printf("\n");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}