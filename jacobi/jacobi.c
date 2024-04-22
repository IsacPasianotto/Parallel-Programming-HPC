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

#include<openacc.h>



/*** function declarations ***/

// save matrix to file
void save_gnuplot( double *M, size_t dim );
void save_gnuplot_parallel( double* M, size_t dimension, size_t local_size, int rank, int size );
// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t dimension );

// return the elapsed time
double seconds( void );

/*** end function declaration ***/

int main(int argc, char* argv[])
{

  // MPI Initialization
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  #ifdef _OPENACC
    // Assign a GPU to each MPI-process
    const int num_devices = acc_get_num_devices(acc_device_nvidia);
    const int device_id = rank % num_devices;
    acc_set_device_num(device_id, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    #ifdef DEBUG_OPENACC
      printf("MPI rank %d of %d has been assigned to GPU %d\n", rank, size, device_id);
    #endif
  #endif


  // timing variables
  double t_start, t_end, increment;

  // indexes for loops
  size_t i, j, it;

  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  // size_t dimension = 0, iterations = 0, row_peek = 0, col_peek = 0;
  size_t dimension = 0, iterations = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  // if(argc != 5)
  // {
  //   fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
  //   return 1;
  // }

  if (argc != 3)
  {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);
  // row_peek = atoi(argv[3]);
  // col_peek = atoi(argv[4]);

  #ifdef VERBOSE
    if (rank == 0)
    {
      printf("matrix size = %zu\n", dimension);
      printf("number of iterations = %zu\n", iterations);
    }
  #endif

  // if((row_peek > dimension) || (col_peek > dimension)){
  //   fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
  //   fprintf(stderr, "Arguments n and m must be smaller than %zu\n", dimension);
  //   return 1;
  // }


  // added to handle MPI communication
  size_t local_size = dimension / size;
  if (rank < (dimension % size))
  {
    local_size++;
  }


  // byte_dimension = sizeof(double) * ( dimension + 2 ) * ( dimension + 2 );
  byte_dimension = sizeof(double) * ( local_size + 2 ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  memset( matrix, 0, byte_dimension );
  memset( matrix_new, 0, byte_dimension );

  //fill initial values
  // for( i = 1; i <= dimension; ++i )
  //   for( j = 1; j <= dimension; ++j )
  //     matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;

  for (i = 1; i <= local_size; ++i)
    for (j = 1; j <= dimension; ++j)
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;


  // set up borders

  // increment = 100.0 / ( dimension + 1 );
  // for( i=1; i <= dimension+1; ++i )
  // {
  //   matrix[ i * ( dimension + 2 ) ] = i * increment;
  //   matrix[ ( ( dimension + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
  //   matrix_new[ i * ( dimension + 2 ) ] = i * increment;
  //   matrix_new[ ( ( dimension + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
  // }


  // need to compute the offset for each block of the matrix
  int n_reminder =  ( (dimension+2)/size ) + 1;
  int n_standard = (dimension+2)/size;
  int diff  = rank - (dimension+2)%size;  // the number of blocks that have n_reminder elements
  int offset = (rank < (dimension+2)%size) ? rank * n_reminder : ( (dimension+2)%size ) * n_reminder + diff * n_standard;

  for (i = 1; i <= local_size; ++i )
  {
    // need to initialize only the vertical borders, the bottom will
    // be initialized only by the process that has the last row
    matrix[ i * ( dimension + 2 ) ] = (offset + i) * increment;
    matrix_new[ ( ( local_size + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
  }

  // the last process has to initialize the bottom border
  if (rank == (size-1) )
  {
    for ( i = 1; i <= dimension+1; ++i )
    {
      matrix[ ( ( local_size + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
      matrix_new[ ( ( local_size + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
    }
  }


  // start algorithm

  // t_start = seconds();
  // for( it = 0; it < iterations; ++it ){
  //
  //   evolve( matrix, matrix_new, dimension );
  //
  //   // swap the pointers
  //   tmp_matrix = matrix;
  //   matrix = matrix_new;
  //   matrix_new = tmp_matrix;
  //
  // }
  // t_end = seconds();


  // Now the situation is a little bit more complex: 
  #pragma acc enter data copyin( matrix[:(dimension+2)*(local_size+2)], matrix_new[:(dimension+2)*(local_size+2)] )
  for (it = 0; it < iterations; ++it)
  {
    
    // Compute the send and recive
    int proc_above = rank > 0 ? rank-1 : MPI_PROC_NULL;
    int proc_below = rank < size-1 ? rank+1 : MPI_PROC_NULL;

    double time = seconds();

    // perform the communication between data in the gpus without passing through the cpu
    #pragma acc host_data use_device(matrix)
    {
      MPI_Sendrecv(matrix + dimension+2, dimension+2, MPI_DOUBLE, proc_above, 0, matrix + (dimension+2)*(local_size+1), dimension+2, MPI_DOUBLE, proc_below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      MPI_Sendrecv(matrix + (dimension+2)*local_size, dimension+2, MPI_DOUBLE, proc_below, 0, matrix, dimension+2, MPI_DOUBLE, proc_above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    
    double time2 = seconds();
    // TODO --> add it to the communication time somehow 


    // Now we can actually evolve the matrix 
    // TODO --> move this part into the "evolve" function under the main 
    #pragma acc parallel loop present(matrix[:(dimension+2)*(local_size+2)], matrix_new[:(dimension+2)*(local_size+2)])
      //  #pragma acc paralel loop collapse(2)
      // {
        for(int i = 1; i <= local_size; ++i)
        {
          for (int j = 1; j <= dimension; ++j)
          {
           matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
             ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
               matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
               matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
               matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
          }
        }
      // }

    // swap the pointers: 
    // The swap is done both in the device and in the host, in order 
    // to preserve the data consistency
    #pragma acc serial present(matrix[:(dimension+2)*(local_size+2)], matrix_new[:(dimension+2)*(local_size+2)])
    {
      double *tmp = matrix;
      matrix = matrix_new;
      matrix_new = tmp;
    }
    // on the ost 
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;

  }  /*   end of iterations loop   */

  // copy the data back to the host and get ready to save the data 
  #pragma acc exit data copyout( matrix[:(dimension+2)*(local_size+2)], matrix_new[:(dimension+2)*(local_size+2)] )

  t_end = seconds();
  


  printf( "\nelapsed time = %f seconds\n", t_end - t_start );
  // printf( "\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[ ( row_peek + 1 ) * ( dimension + 2 ) + ( col_peek + 1 ) ] );

  
  // Replace the save_gnuplot function with the same function which uses MPI_IO
  // save_gnuplot( matrix, dimension );
  save_gnuplot_parallel( matrix, dimension, local_size, rank, size );

  free( matrix );
  free( matrix_new );



  MPI_Finalize();
  return 0;
}

void evolve( double * matrix, double *matrix_new, size_t dimension ){

  size_t i , j;

  //This will be a row dominant program.
  for( i = 1 ; i <= dimension; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) *
                                                    ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] +
                                                      matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] +
                                                      matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] +
                                                      matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] );
}

void save_gnuplot( double *M, size_t dimension )
{

  size_t i , j;
  const double h = 0.1;
  FILE *file;

  file = fopen( "solution.dat", "w" );

  for( i = 0; i < dimension + 2; ++i )
    for( j = 0; j < dimension + 2; ++j )
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( dimension + 2 ) ) + j ] );

  fclose( file );

}


void save_gnuplot_parallel( double* M, size_t dimension, size_t local_size, int rank, int size ) 
{

  // use MPI_IO to save the data in parallel
  MPI_File fh;
  MPI_Offset offset;
  MPI_Status status;

  // open the file
  MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  // compute the offset
  offset = (rank * local_size) * (dimension + 2) * sizeof(double);
  // write the data
  MPI_File_write_at(fh, offset, M, local_size * (dimension + 2), MPI_DOUBLE, &status);
  // close the file
  MPI_File_close(&fh);
}



// A Simple timer for measuring the walltime
double seconds()
{

  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}
