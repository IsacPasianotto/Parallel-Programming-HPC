#include <stdio.h>
#include <sys/time.h>

#include "stopwatch.h"


/*   This is the timer function provided by the teacher.
     Students were suggested to use this function to measure the time of their programs. */
double get_time_in_seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *) 0 );
  sec = tmp.tv_sec + ( (double) tmp.tv_usec ) / 1000000.0;
  return sec;
}

void record_time(double* time_records, int* index)
{
  time_records[*index] = get_time_in_seconds();
  (*index)++;
}

void print_time_records(double* time_records, int rank, int size, const char* program_type)
{
  printf("%f,%d,%d,%s,%s\n", time_records[1] - time_records[0], rank, size, program_type, "init-A-B-C");
  for (int i = 0; i < size; i++)
  {
    printf("%f,%d,%d,%s,%s\n", time_records[3 + 5*i] - time_records[2 + 5*i], rank, size, program_type, "compute-col-block-B");
    printf("%f,%d,%d,%s,%s\n", time_records[4 + 5*i] - time_records[3 + 5*i], rank, size, program_type, "get-ready-allgatherv");
    printf("%f,%d,%d,%s,%s\n", time_records[5 + 5*i] - time_records[4 + 5*i], rank, size, program_type, "perform-allgatherv");
    printf("%f,%d,%d,%s,%s\n", time_records[6 + 5*i] - time_records[5 + 5*i], rank, size, program_type, "compute-block-C-result");
  }
}

void print_time_records_cuda(double* time_records, int rank, int size, const char* program_type)
{
  printf("%f,%d,%d,%s,%s\n", time_records[1] - time_records[0], rank, size, program_type, "init-A-B-C");
  printf("%f,%d,%d,%s,%s\n", time_records[2] - time_records[1], rank, size, program_type,"Copy-A-and-C-to-device");
  for (int i = 0; i < size; i++)
  {
    printf("%f,%d,%d,%s,%s\n", time_records[4+ 9*i] - time_records[3 + 9*i], rank, size, program_type, "compute-col-block-B");
    printf("%f,%d,%d,%s,%s\n", time_records[5+ 9*i] - time_records[4 + 9*i], rank, size, program_type, "get-ready-allgatherv");
    printf("%f,%d,%d,%s,%s\n", time_records[6+ 9*i] - time_records[5 + 9*i], rank, size, program_type, "perform-allgatherv");
    printf("%f,%d,%d,%s,%s\n", time_records[8+ 9*i] - time_records[7 + 9*i], rank, size, program_type, "copy-column-B-cpu-to-gpu");
    printf("%f,%d,%d,%s,%s\n", time_records[9+ 9*i] - time_records[8 + 9*i], rank, size, program_type, "compute-block-C-result");
    printf("%f,%d,%d,%s,%s\n", time_records[10+ 9*i] - time_records[9 + 9*i], rank, size, program_type, "copy-block-C-gpu-to-cpu");
    printf("%f,%d,%d,%s,%s\n", time_records[11+ 9*i] - time_records[10 + 9*i], rank, size, program_type, "move-block-C-to-global-C");
  }
}
