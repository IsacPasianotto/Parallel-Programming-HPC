#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#include "init.h"


void init_local_matrix (double* M, long int n_elements)
{
  #pragma omp parallel for
  for (long int i = 0; i < n_elements; i++) {
    M[i] = (double) rand()/1000000;
  }
}
