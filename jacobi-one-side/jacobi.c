/*---------------------------------------------*
 | file: jacobi.c                              |
 | author: Ivan Girotto  (Prof. of the course) |
 | edited by: Isac Pasianotto                  |
 | date: 2024-04                               |
 | context: exam of "Parallel programming for  |
 |      HPC". Msc Course in DSSC               |
 | description: Jacobi method for solving a    |
 |      Laplace equation, implementation of    |
 |      the exercise using the RMA model of    |
 |      MPI.                                   |
 *---------------------------------------------*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

/*** function declarations ***/
// return the elapsed time
double seconds(void);
// mpi-needs functions
int calculate_local_size(int tot_col, int size, int rank);
void set_recvcout(int* recvcount, int size, int N);
void set_displacement(int* displacement, const int* recvcount, int size);

/*** end function declaration ***/

/***    main     ***/

int main(int argc, char* argv[])
{

  /**   MPI initialization   **/
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /**   end MPI initialization   **/


  /**   Variables initialization   **/

  // -- declaration
  double increment;
  #ifdef STOPWATCH
    double communication_time=0, compute_time=0;  // timing variables
    double start_init, end_init;
  #endif

  size_t i, j, it;                             // indexes for loops
  double *matrix, *matrix_new, *tmp_matrix;    // initialize matrix
  size_t dimension = 0, iterations = 0;
  size_t byte_dimension = 0;

  if (argc != 3)
  {
    fprintf(stderr, "\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  size_t local_size = dimension / size;
  if (rank < (dimension % size))
  {
    local_size++;
  }

  // -- allocation
  byte_dimension = sizeof(double) * (local_size + 2) * (dimension + 2);
  matrix = (double*)malloc(byte_dimension);
  matrix_new = (double*)malloc(byte_dimension);

  memset(matrix, 0, byte_dimension);
  memset(matrix_new, 0, byte_dimension);

  // -- initialization
  #ifdef STOPWATCH
    start_init = seconds();
  #endif

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
  int diff = (dimension + 2) % size;  // The number of blocks that have n_reminder elements
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
  #ifdef STOPWATCH
    end_init = seconds();
  #endif
  /** end of variable initialization **/

  /** start the actual algorithm  **/
  for (it = 0; it < iterations; ++it)
  {
    // send up, recv bottom
    int proc_above = (rank - 1) >= 0 ? rank - 1 : MPI_PROC_NULL;
    int proc_below = (rank + 1) < size ? rank + 1 : MPI_PROC_NULL;
    double time = seconds();

  #ifndef ONESIDE
    MPI_Sendrecv(matrix + (dimension + 2), dimension + 2, MPI_DOUBLE, proc_above, 0, matrix + (dimension + 2) * (local_size + 1), dimension + 2, MPI_DOUBLE, proc_below, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(matrix + (dimension + 2) * local_size, dimension + 2, MPI_DOUBLE, proc_below, 2, matrix, dimension + 2, MPI_DOUBLE, proc_above, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  #else     //-- Two-sided communication
    #ifdef ONEWIN     // open one window (try to reduce overhead)
      MPI_Win win;
      MPI_Win_create(matrix, byte_dimension, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
      MPI_Win_fence(0, win);
      #ifdef PUT
        MPI_Put(matrix + (dimension + 2), dimension + 2, MPI_DOUBLE, proc_above, (dimension + 2) * (local_size + 1), dimension + 2, MPI_DOUBLE, win);
        MPI_Put(matrix + (dimension + 2) * local_size, dimension + 2, MPI_DOUBLE, proc_below, 0, dimension + 2, MPI_DOUBLE, win);
      #else //  ifdef GET
        MPI_Get(matrix + (dimension + 2) * (local_size + 1), dimension + 2, MPI_DOUBLE, proc_below, dimension + 2, dimension + 2, MPI_DOUBLE, win);
        MPI_Get(matrix, dimension + 2, MPI_DOUBLE, proc_above, (dimension + 2) * local_size, dimension + 2, MPI_DOUBLE, win);
      #endif  // end of ifdef GET
      MPI_Win_fence(0, win);
      MPI_Win_free(&win);
    #else   // end of one window condition ;   -- Two windows
      double* ghost_up = matrix;
      double* ghost_down = matrix + (local_size + 1) * (dimension + 2);
      double* first_row_point = matrix + (dimension + 2);
      double* last_row_point = matrix + (dimension + 2) * local_size;

      MPI_Win ghost_up_win, ghost_down_win;

      #ifdef PUT
        MPI_Win_create(ghost_up, dimension * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghost_up_win);
        MPI_Win_create(ghost_down, dimension * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghost_down_win);
      #else //  ifdef GET
        MPI_Win_create(first_row_point, dimension * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghost_up_win);
        MPI_Win_create(last_row_point, dimension * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &ghost_down_win);
      #endif

      MPI_Win_fence(0, ghost_up_win);
      MPI_Win_fence(0, ghost_down_win);

      #ifdef PUT
        MPI_Put(first_row_point, dimension, MPI_DOUBLE, proc_above, 0, dimension, MPI_DOUBLE, ghost_down_win);
        MPI_Put(last_row_point, dimension, MPI_DOUBLE, proc_below, 0, dimension, MPI_DOUBLE, ghost_up_win);
      #else
        MPI_Get(ghost_down, dimension, MPI_DOUBLE, proc_below, 0, dimension, MPI_DOUBLE, ghost_up_win);
        MPI_Get(ghost_up, dimension, MPI_DOUBLE, proc_above, 0, dimension, MPI_DOUBLE, ghost_down_win);
      #endif

      MPI_Win_fence(0, ghost_down_win);
      MPI_Win_fence(0, ghost_up_win);

      MPI_Win_free(&ghost_up_win);
      MPI_Win_free(&ghost_down_win);

    #endif  // end of two window condition
  #endif  // end of ONESIDE condition

    #ifdef STOPWATCH
      communication_time += seconds() - time;
      time = seconds();
    #endif

    /** evolve the matrix **/

    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i <= local_size; ++i)
    {
      for (size_t j = 1; j <= dimension; ++j)
      {
        matrix_new[(i * (dimension + 2)) + j] = 0.25 * (matrix[((i - 1) * (dimension + 2)) + j] + matrix[(i * (dimension + 2)) + (j + 1)] + matrix[((i + 1) * (dimension + 2)) + j] + matrix[(i * (dimension + 2)) + (j - 1)]);
      }
    }
    #ifdef STOPWATCH
      compute_time += seconds() - time;
    #endif

    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  } /* end of loop over iteration iterations */
  MPI_Barrier(MPI_COMM_WORLD);
  /**   end of the matrix evolution   **/

  /**   Save the results   **/
    MPI_File fh;
    MPI_Offset file_disp = 0;
    MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    // set the offset
    for( int i=0; i<rank; ++i)
    {
      file_disp+=calculate_local_size(dimension,size,i)*(dimension+2)*sizeof(double);
    }
    MPI_File_write_at(fh, file_disp, matrix, (dimension+2)*(local_size+2), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  /** end of saving the results **/

  /**  Print the times  **/
  #ifdef STOPWATCH
    // time, rank, size, what
    printf("%.10f,%d,%d,%s\n", end_init - start_init, rank, size, "matrix-initialization");
    printf("%.10f,%d,%d,%s\n", communication_time, rank, size, "mpi-comm");
    printf("%.10f,%d,%d,%s\n", compute_time, rank, size, "computation");
  #endif
  /**  End of printing time **/

  /**   Finalize the program   **/

  free( matrix );
  free( matrix_new );

  MPI_Finalize();
  return 0;
}

/***  end of main ***/


/*** function definitions ***/
double seconds()        // A Simple timer for measuring the walltime
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

int calculate_local_size(int tot_col,int size,int rank)   //calculate how many row belong to the current rank
{
  return (rank < tot_col % size) ? tot_col/size +1 : tot_col/size;
}

void set_recvcout(int* recvcount, int size,int N)       //set the recv_count array
{
  for(int p=0;p<size;++p)
  {
    recvcount[p]=calculate_local_size(N,size,p)*(N+2);
  }
  recvcount[0]+=N+2;
  recvcount[size-1]+=N+2;
}

void set_displacement(int* displacement,const int* recvcount,int size)    //calculate the displacement array using the recv_count array
{
  displacement[0] = 0;
  for (int p = 1; p < size; ++p)
  {
    displacement[p] = displacement[p - 1] + recvcount[p - 1];
  }
}
/***  end of function declaration   ***/