#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

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
