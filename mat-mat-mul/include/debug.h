#ifndef INIT_DEBUG
#define INIT_DEBUG
  void debug_init_local_matrix(double* A, double* B, long int N, long int local_size, int rank, int size);
#endif

#ifndef DEBUG_COLUMN_BLOCK
#define DEBUG_COLUMN_BLOCK
  void debug_col_block(double* B, double* local_block, long int N, long int local_size, int rank, int iter, int* all_sizes);
#endif