/*---------------------------------------------*
 | file: jacobi.c                              |
 | author: Ivan Girotto  (Prof. of the course) |
 | edited by: Isac Pasianotto                  |
 | date: 2024-05                               |
 | context: exam of "Parallel programming for  |
 |      HPC". Msc Course in DSSC               |
 | description: Jacobi method for solving a    |
 |      Laplace equation, using the one-side   |
 |      communication MPI paradigm.            |
 *---------------------------------------------*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

/*** function declarations ***/

// save matrix to .dat file in order to render with gnuplot
void save_gnuplot(double *M, size_t dim);
// return the elapsed time
double seconds(void);
// mpi-needs functions
int calculate_local_size(int tot_col, int size, int rank);
void set_recvcout(int *recvcount, int size, int N);
void set_displacement(int *displacement, const int *recvcount, int size);

/*** end function declaration ***/

/*** main ***/

int main(int argc, char *argv[])
{
  /** MPI initialization **/
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /** end MPI initialization **/

  /** Variables initialization **/
  // -- declaration
  double increment; // timing variables
  double communication_time = 0, compute_time = 0;
  double start_time = 0, end_time;

  size_t i, j, it;                  // indexes for loops
  double *matrix, *matrix_new, *tmp_matrix; // initialize matrix
  size_t dimension = 0, iterations = 0;
  size_t byte_dimension = 0;
  // send up, recv bottom
  int send_to;
  int recv_from;

  if (argc != 3)
  {
    fprintf(stderr, "\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  #ifdef VERBOSE
    if (rank == 0)
      {
          printf("matrix size = %zu\n", dimension);
          printf("number of iterations = %zu\n", iterations);
      }
  #endif

  size_t local_size = dimension / size;
  if (rank < (dimension % size))
  {
    local_size++;
  }

  // -- allocation
  byte_dimension = sizeof(double) * (local_size + 2) * (dimension + 2);
  matrix = (double *)malloc(byte_dimension);
  matrix_new = (double *)malloc(byte_dimension);

  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  // -- initialization
  start_time = seconds();

  // fill initial values
  #pragma omp parallel for collapse(2)
  for (i = 1; i <= local_size; ++i)
  {
    for (j = 1; j <= dimension; ++j)
    {
      matrix[(i * (dimension + 2)) + j] = 0.5;
    }
  }

  increment = 100.0 / (dimension + 1);
  // Compute the offset
  int size_std = (dimension + 2) / size;
  int size_reminder = size_std + 1;
  int diff = (dimension + 2) % size; // The number of blocks that have n_reminder elements
  int offset = (rank < diff) ? rank * size_reminder : diff * size_std + (rank - diff) * size_std;

  // need to initialize only the vertical borders
  for (i = 1; i <= local_size + 1; ++i)
  {
    matrix[i * (dimension + 2)] = (i + offset) * increment;
    matrix_new[i * (dimension + 2)] = (i + offset) * increment;
  }

  // The last process init also the horizontal border
  if (rank == (size - 1))
  {
    for (i = 1; i <= dimension + 1; ++i)
    {
      matrix[((local_size + 1) * (dimension + 2)) + (dimension + 1 - i)] = i * increment;
      matrix_new[((local_size + 1) * (dimension + 2)) + (dimension + 1 - i)] = i * increment;
    }
  }
  end_time = seconds();

  // compute upper and lower bounds for the communication
  send_to= (rank - 1) >= 0 ? rank - 1 : MPI_PROC_NULL;
  recv_from = (rank + 1) < size ? rank + 1 : MPI_PROC_NULL;

  /** end of variable initialization **/

  /** start the actual algorithm **/
  for (it = 0; it < iterations; ++it)
  {
    double comm_time_start = seconds();

    MPI_Sendrecv(matrix + (dimension + 2), dimension + 2, MPI_DOUBLE, send_to, 0,
                 matrix + (dimension + 2) * (local_size + 1), dimension + 2, MPI_DOUBLE, recv_from, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(matrix + (dimension + 2) * local_size, dimension + 2, MPI_DOUBLE, recv_from, 0,
                 matrix, dimension + 2, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    communication_time += seconds() - comm_time_start;

    double compute_time_start = seconds();

    /** evolve the matrix **/
    #pragma omp parallel for collapse(2)
    for (i = 1; i <= local_size; ++i)
    {
      for (j = 1; j <= dimension; ++j)
      {
        matrix_new[(i * (dimension + 2)) + j] = 0.25 * (
            matrix[((i - 1) * (dimension + 2)) + j] +
            matrix[(i * (dimension + 2)) + (j + 1)] +
            matrix[((i + 1) * (dimension + 2)) + j] +
            matrix[(i * (dimension + 2)) + (j - 1)]);
      }
    }

    compute_time += seconds() - compute_time_start;

    // swap pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  } /* end of loop over iteration iterations */

  /** end of the matrix evolution **/

  /** Save the results **/
  MPI_Barrier(MPI_COMM_WORLD);

  // use MPI-IO to write the file in parallel.
  MPI_File fh;
  MPI_Offset file_disp = 0;
  MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  // set the offset
  for (int i = 0; i < rank; ++i)
  {
    file_disp += calculate_local_size(dimension, size, i) * (dimension + 2) * sizeof(double);
  }
  MPI_File_write_at(fh, file_disp, matrix, (dimension + 2) * (local_size + 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  /** end of saving the results **/

  /** Print the times **/
  #ifdef STOPWATCH
    // time, rank, size, what
    printf("%.10f,%d,%d,%s\n", end_time - start_time, rank, size, "matrix-initialization");
    #ifdef _OPENACC
    printf("%.10f,%d,%d,%s\n", copyin_end - copyin_start, rank, size, "copy-matrix-cpu-to-gpu");
    printf("%.10f,%d,%d,%s\n", copyout_end - copyout_start, rank, size, "copy-matrix-gpu-to-cpu");
    #endif
    printf("%.10f,%d,%d,%s\n", communication_time, rank, size, "mpi-send-rec");
    printf("%.10f,%d,%d,%s\n", compute_time, rank, size, "computation");
  #endif
  /** End of printing time **/

  /** Finalize the program **/
  free(matrix);
  free(matrix_new);

  MPI_Finalize();
  return 0;
}

/*** end of main ***/

/*** function definitions ***/

void save_gnuplot(double *M, size_t dimension)
{
  size_t i, j;
  const double h = 0.1;
  FILE *file = fopen("solution.dat", "w");
  for (i = 0; i <= dimension + 1; ++i)
  {
    for (j = 0; j <= dimension + 1; ++j)
    {
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[(i * (dimension + 2)) + j]);
    }
  }
  fclose(file);
}

double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday(&tmp, (struct timezone *)0);
  sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;
  return sec;
}

int calculate_local_size(int tot_col, int size, int rank)
{
  return (rank < tot_col % size) ? tot_col / size + 1 : tot_col / size;
}

void set_recvcout(int *recvcount, int size, int N)
{
  for (int p = 0; p < size; ++p)
  {
    recvcount[p] = calculate_local_size(N, size, p) * (N + 2);
  }
  recvcount[0] += N + 2;
  recvcount[size - 1] += N + 2;
}

void set_displacement(int *displacement, const int *recvcount, int size)
{
  displacement[0] = 0;
  for (int p = 1; p < size; ++p)
  {
    displacement[p] = displacement[p - 1] + recvcount[p - 1];
  }
}
/*** end of function definitions ***/
