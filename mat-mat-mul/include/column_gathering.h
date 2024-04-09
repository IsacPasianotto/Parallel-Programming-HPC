#ifndef BUILD_COLUMN_BLOCK
#define BUILD_COLUMN_BLOCK
  void build_column_block(double* local_block, double* B, long int N, long int local_size, int size, int iter, int* all_sizes);
#endif

#ifndef COMPUTE_RECEIVE_COUNTS
#define COMPUTE_RECEIVE_COUNTS
  void compute_receive_counts(int* revcounts, int* all_sizes, int size, int iter);
#endif

#ifndef COMPUTE_DISPLACEMENTS
#define COMPUTE_DISPLACEMENTS
  void compute_displacements(int* displs, int* revcounts, int size);
#endif