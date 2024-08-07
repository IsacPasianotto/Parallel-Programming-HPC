/*-----------------------------------------------*
| file: init.c.c                                 |
| author: Isac Pasianotto                        |
| date: 2024-04                                  |
| context: exam of "Parallel programming for     |
|          HPC". Msc Course in DSSC              |
| description: function that randomly initialize |
|    A matrix. In case of SMALL_INTS, the matrix |
|    is initialized to integers to help debugging|
*------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "init.h"


void init_local_matrix (double* M, long int n_elements)
{
  #pragma omp parallel for
  for (long int i = 0; i < n_elements; i++)
  {
    M[i] = (double) rand()/1000000;
#ifdef SMALL_INTS
    M[i] = (double) i;
#endif
    // 50% of chance to be negative -> help to avoid overflow
    if (rand() % 2 == 0)
    {
      M[i] *= -1;
    }
  }
}

void init_local_matrix_thread_safe(double* M, long int n_elements)
{
  #pragma omp parallel
  {
    unsigned int seed = omp_get_thread_num();
    #pragma omp for
    for (long int i = 0; i < n_elements; i++)
    {
      M[i] = (double) rand_r(&seed)/1000000;
      if (rand_r(&seed) % 2 == 0)
      {
        M[i] *= -1;
      }
    }
  }
}