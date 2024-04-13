#ifndef ASSIGN_GPU_TO_PROCESS
#define ASSIGN_GPU_TO_PROCESS
  void assign_gpu_to_process(int rank);
#endif

#ifndef GET_READY_ON_GPU
#define GET_READY_ON_GPU
  void get_ready_on_gpu(double* A, double* C, double* d_A, double* d_C, long int N, long int local_size, int rank, int size, double* time_records, int* time_counter)
#endif

#ifndef COMPUTE_BLOCK_RESULT_CUDA
#define COMPUTE_BLOCK_RESULT_CUDA
  void compute_block_result_cuda(double* d_A, double* d_C, double* buffer, long int N, long int local_size, int* all_sizes, int size, int iter, double* time_records, int* time_counter);
#endif

#ifndef CUDA_COPY_BLOCK_TO_GLOBAL_C
#define CUDA_COPY_BLOCK_TO_GLOBAL_C
  __global__ void cuda_copy_block_to_global_c(double* d_C, double* local_C_block, long int N, long int local_size, int* all_sizes, int size, int iter);
#endif

#ifndef FREE_GPU_MEMORY_LOOP
#define FREE_GPU_MEMORY_LOOP
  void free_gpu_memory_loop(int* sendcounts, int* displs, double* local_C_block, double* buffer, cublasHandle_t handle, double* device_C_block, double* device_B_buffer);
#endif

#ifndef FREE_GPU_MEMORY
#define FREE_GPU_MEMORY
  void free_gpu_memory(double* A, double* B, double* C, int* all_sizes, double* time_records);
#endif