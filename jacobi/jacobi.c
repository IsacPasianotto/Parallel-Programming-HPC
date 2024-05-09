/*---------------------------------------------*
 | file: jacobi.c                              |
 | author: Ivan Girotto  (Prof. of the course) |
 | edited by: Isac Pasianotto                  |
 | date: 2024-04                               |
 | context: exam of "Parallel programming for  |
 |      HPC". Msc Course in DSSC               |
 | description: Jacobi method for solving a    |
 |      Laplace equation, ported on GPU using  |
 |      openacc and distributed using MPI      |
 *---------------------------------------------*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#ifdef _OPENACC
#include <accel.h>
#endif

/*** function declarations ***/

// save matrix to  .dat file in order to render with gnuplot
void save_gnuplot( double *M, size_t dim );
// return the elapsed time
double seconds( void );
// mpi-needs functions
int calculate_local_size(int tot_col,int size,int rank);
void set_recvcout(int* recvcount, int size,int N);
void set_displacement(int* displacement,const int* recvcount,int size);

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

  /**   OpenACC initialization   **/
  #ifdef _OPENACC
    // Assign a GPU to each MPI-process
    const int num_devices = acc_get_num_devices(acc_device_nvidia);
    const int device_id = rank % num_devices;
    acc_set_device_num(device_id, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    #ifdef DEBUG_OPENACC
      printf("MPI rank %d of %d has been assigned to GPU %d\n", rank, size, device_id);
    #endif// OpenACC call
  #endif

  /**   end OpenACC initialization   **/

  /**   Variables initialization   **/

  // -- declaration
  double increment;                            // timing variables
  double communication_time=0,compute_time=0;
  double start_init, end_init;
  #ifdef _OPENACC
    double copyin_start, copyin_end, copyout_start, copyout_end;
  #endif
  size_t i, j, it;                             // indexes for loops
  double *matrix, *matrix_new, *tmp_matrix;    // initialize matrix
  size_t dimension = 0, iterations = 0;
  size_t byte_dimension = 0;
  #ifndef FASTOUTPUT
    int* recvcount=NULL;                         // MPI variables
    int* displacement=NULL;
    double* matrix_final=NULL;
  #endif
  if(argc != 3)
  {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);

  #ifdef VERBOSE
    if(rank==0)
    {
      printf("matrix size = %zu\n", dimension);
      printf("number of iterations = %zu\n", iterations);
    }
  #endif

  size_t local_size = dimension/size;
  if(rank < (dimension % size))
  {
    local_size++;
  }

  // -- allocation
  byte_dimension = sizeof(double) * ( local_size + 2 ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  memset( matrix, 0, byte_dimension );
  memset( matrix_new, 0, byte_dimension );

  // -- initialization
  start_init = seconds();

  //fill initial values
  #pragma omp parallel for collapse(2)
  for( i = 1; i <= local_size; ++i )
  {
    for (j = 1; j <= dimension; ++j)
    {
      matrix[(i * (dimension + 2)) + j] = 0.5;
    }
  }

  increment = 100.0 / ( dimension + 1 );
  // Compute the offset
  int size_std = (dimension + 2) / size;
  int size_reminder = size_std + 1;
  int diff = (dimension + 2) % size;  // The number of blocks that have n_reminder elements
  int offset = (rank < diff) ? rank * size_reminder : diff * size_std + (rank - diff) * size_std;

  // need to initialize only the vertical borders
  for( i=1; i <= local_size+1; ++i )
  {
    matrix[ i * ( dimension + 2 ) ] = (i+offset) * increment;
    matrix_new[ i * ( dimension + 2 ) ] = (i+offset)  * increment;
  }

  //The last process init also the horizontal border
  if(rank==(size-1))
  {
    for (i = 1; i <= dimension + 1; ++i)
    {
      matrix[((local_size + 1) * (dimension + 2)) + (dimension + 1 - i)] = i * increment;
      matrix_new[((local_size + 1) * (dimension + 2)) + (dimension + 1 - i)] = i * increment;
    }
  }
  end_init = seconds();
  /** end of variable initialization **/

  /** start the actual algorithm  **/

  #ifdef _OPENACC
    copyin_start = seconds();
  #endif
  #pragma acc enter data copyin(matrix[:(dimension+2)*(local_size+2)],matrix_new[:(dimension+2)*(local_size+2)])
  {
    #ifdef _OPENACC
      copyin_end = seconds();
    #endif
    for (it = 0; it < iterations; ++it)
    {
      //send up,recv bottom
      int send_to = (rank - 1) >= 0 ? rank - 1 : MPI_PROC_NULL;
      int recv_from = (rank + 1) < size ? rank + 1 : MPI_PROC_NULL;
      double time = seconds();

      #pragma acc host_data use_device(matrix)
      {
        MPI_Sendrecv(matrix + (dimension + 2), dimension + 2, MPI_DOUBLE, send_to, 0, matrix + (dimension+2) * (local_size + 1), dimension+2, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(matrix + (dimension + 2) * local_size, dimension + 2, MPI_DOUBLE, recv_from, 0, matrix, dimension+2, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      communication_time += seconds() - time;

      time = seconds();

      /** evolve the matrix **/

      #pragma acc  data present(matrix[:(dimension+2)*(local_size+2)], matrix_new[:(dimension+2)*(local_size+2)])
      {
        #ifdef _OPENACC
          #pragma acc parallel loop collapse(2)
        #else
          #pragma omp parallel for collapse(2)
        #endif
          for (int i = 1; i <= local_size; ++i)
          {
            for (int j = 1; j <= dimension; ++j)
            {
              matrix_new[(i * (dimension + 2)) + j] = (0.25) *
                                                      (matrix[((i - 1) * (dimension + 2)) + j] +
                                                       matrix[(i * (dimension + 2)) + (j + 1)] +
                                                       matrix[((i + 1) * (dimension + 2)) + j] +
                                                       matrix[(i * (dimension + 2)) + (j - 1)]);

            }
          }
      }
      compute_time += seconds() - time;

      #ifdef _OPENACC
        //swap the pointers on the device
        #pragma acc serial present(matrix[:(dimension+2)*(local_size+2)],matrix_new[:(dimension+2)*(local_size+2)])
        {
          double* tmp_matrix = matrix;
          matrix = matrix_new;
          matrix_new = tmp_matrix;
        }
      #endif
      // swap pointer on the host (to do even if using openacc to preserve data consistency)
      tmp_matrix = matrix;
      matrix = matrix_new;
      matrix_new = tmp_matrix;

    } /* end of loop over iteration iterations */
  } /* end of acc data region */

  /**   end of the matrix evolution   **/

  /**   Save the results   **/

  //Back to the host
  #ifdef _OPENACC
    copyout_start = seconds();
  #endif
  #pragma acc exit data copyout(matrix[:(dimension+2)*(local_size+2)],matrix_new[:(dimension+2)*(local_size+2)])
  #ifdef _OPENACC
    copyout_end = seconds();
  #endif
  MPI_Barrier(MPI_COMM_WORLD);

  #ifdef FASTOUTPUT
    /*
     * Optimal solution, use MPI-IO to write the file in parallel.
     */
    MPI_File fh;
    MPI_Offset file_disp = 0;
    MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    // set the offset
    for(int i=0;i<rank;++i)
    {
      file_disp+=calculate_local_size(dimension,size,i)*(dimension+2)*sizeof(double);
    }
    MPI_File_write_at(fh, file_disp, matrix, (dimension+2)*(local_size+2), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
  #else
    /*
     * Sub-optimal solution, needed only to plot with the already-give gnu-plot script
     * In this case the matrix is gathered on the root process and saved in a file by it.
     * This is not the best solution since it involves:
     * 0. a lot of avoidable communication
     * 1. a lot of memory allocation
     * 2. Writing the file in serial
     */
    if (rank==0)
    {
      recvcount=malloc(size*sizeof(int));
      displacement=malloc(size*sizeof(int));
      matrix_final=malloc((dimension+2)*(dimension+2)*sizeof(double));
      set_recvcout(recvcount,size,dimension);
      set_displacement(displacement,recvcount,size);
    }
    if(rank==0)
    {
      MPI_Gatherv(matrix_new, (dimension+2)*(local_size+1), MPI_DOUBLE, matrix_final, recvcount, displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if(rank == (size-1))
    {
      MPI_Gatherv(matrix_new+(dimension+2),  (dimension+2)*(local_size+1), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if ( !(rank==0 || rank == (size-1)) )
    {
      MPI_Gatherv(matrix_new+(dimension+2),  (dimension+2)*(local_size), MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if(rank==0)
    {
      save_gnuplot( matrix_final, dimension );
    }
  #endif  // enf if not FASTOUTPUT

  /** end of saving the results **/

  /**  Print the times  **/

#ifdef STOPWATCH
  // time, rank, size, what
    printf("%.10f,%d,%d,%s\n", end_init - start_init, rank, size, "matrix-initialization");
    #ifdef _OPENACC
      printf("%.10f,%d,%d,%s\n", copyin_end - copyin_start, rank, size, "copy-matrix-cpu-to-gpu");
      printf("%.10f,%d,%d,%s\n", copyout_end - copyout_start, rank, size, "copy-matrix-gpu-to-cpu");
    #endif
    printf("%.10f,%d,%d,%s\n", communication_time, rank, size, "mpi-send-rec");
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
void save_gnuplot( double *M, size_t dimension)
{
  size_t i , j;
  const double h = 0.1;
  FILE *file;
  file = fopen( "solution.dat", "w" );
  for( i = 0; i < dimension + 2; ++i )
  {
    for (j = 0; j < dimension + 2; ++j)
    {
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[(i * (dimension + 2)) + j]);
    }
  }
  fclose( file );
}

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
  displacement[0]=0;
  for(int p=1;p<size;++p)
  {
    displacement[p] = displacement[p - 1] + recvcount[p - 1];
  }
}

/***  end of function declaration   ***/
