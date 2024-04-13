#ifndef GET_TIME_IN_SECONDS
#define GET_TIME_IN_SECONDS
  double get_time_in_seconds();
#endif

#ifndef RECORD_TIME
#define RECORD_TIME
  void record_time(double* time_records, int* index);
#endif

#ifndef PRINT_TIME_RECORDS
#define PRINT_TIME_RECORDS
  void print_time_records(double* time_records, int rank, int size, const char* program_type);
#endif

#ifdef CUDA
#ifndef PRINT_TIME_RECORDS_CUDA
#define PRINT_TIME_RECORDS_CUDA
 void print_time_records_cuda(double* time_records, int rank, int size, const char* program_type);
#endif
#endif