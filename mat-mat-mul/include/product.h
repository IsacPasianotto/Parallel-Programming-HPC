#ifndef COMPUTE_BLOCK_RESULT_NAIVE
#define COMPUTE_BLOCK_RESULT_NAIVE
void compute_block_result_naive (double* local_C_block, double* A, double* buffer, long int N, long int local_size, int* all_sizes, int iter);
#endif

#ifndef COPY_BLOCK_TO_GLOBAL_C
#define COPY_BLOCK_TO_GLOBAL_C
void copy_block_to_global_C (double* C, double* local_C_block, long int N, long int local_size, int* all_sizes,  int size, int iter);
#endif
