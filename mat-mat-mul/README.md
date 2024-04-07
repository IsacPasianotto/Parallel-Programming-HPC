:> [!WARNING]
> REAMRK: at the moment this readme are just the note that I took during the lecture in which professor assigned the exercise. 
> It is going to be re-written as soon as possible


***Remark***: This exercise will be part of the exam.

We are going to parallelize a code that implements the matrxi-matrix multiplication distributely 

$$
A \times B = C
$$

where $A$ and $B$ are matrices and $C$ is the result of the multiplication..

$$
C_{ij} = \sum_{k=0}^{N} A_{ik} \times B_{kj}
$$


Normally in code this is a three time nested loop: 

```c
for (int i = 0; i < N; i++)
{
    for (int j = 0; j < N; j ++)
    {
        for (int k = 0; k < N; k++)
        {
            C[i, j] += A[i, k] * B[k, j];
        }
    }
}
```


But now we are in a setting with the data are distributed among different processes...
There are many algorithms, the one we are going to implement is not the fastes in the world, but is a good compromise between simplicity of implementation and performance.


Instead of gathering one single column and computing one single element of $C$ for all gather, we can gather a block of columns (size: $N \times n_{loc}$).

In this way instead of doing an allghater for each column, the total of allgather operation is reduced to $N/n_{loc} = $ number of processing elements.


Of course this algorithm works better if the number of processing elements is much smaller than the size of the matrix $N$.


note: $T_{communication} = T_{initialization} + T_{data\_transfer}$. 


**Unfortunatly** there is a big complication: at the end of the day, every local processes has to compute locally his local portion of C (local parto fo B times all the portion of B).  The elements we are selecting are not contiguos in memory.


```
P0              B
    +--------+------+--------+             +-------+
    |        |\\\\\\|        |     -->     |\\\\\\\|
    |        |\\\\\\|        |             |\\\\\\\|
    +--------+------+--------+             +-------+

P1              B
    +--------+------+--------+             +-------+
    |        |\\\\\\|        |     -->     |\\\\\\\|
    |        |\\\\\\|        |             |\\\\\\\|
    +--------+------+--------+             +-------+



Pn              B
    +--------+------+--------+             +-------+
    |        |\\\\\\|        |     -->     |\\\\\\\|
    |        |\\\\\\|        |             |\\\\\\\|
    +--------+------+--------+             +-------+

```



- - -  

Steps in pseudocode:

```

t0
Allocation of distributed data
t1
Initialization of A and B (randomly)
t2
for (count = 0; c < Npes< coutn ++){
    t3
    create_block (B, block)             // create a block of columns
    t4
    ALL_GATHER(block, n_loc*n_loc, ...) // gather the block of columns, B_colum is the receiving buffer
    t5
    MAT_MUL(A, B_colum, C_loc)          // compute the local portion of C
    t6
    
}
print C // to check the result, to print it we have to collect all the local portions of C before
```




```
                A                   B_column                    C 
P_x     +-------------------+      +--------+     *-  -  -  -  +--------+ -  -  -  -  - +
        |                   |  x   |\\\\\\\\|     |            |\\\\\\\\|               |
        |                   |      |\\\\\\\\|     |            |\\\\\\\\|               |
        +-------------------+      +--------+     *-  -  -  -  +--------+ -  -  -  -  - +
```




We are required to produce a stacked chart about the time spent in the different parts of the code to compare the time of the parallel portion of the code with the serial one.


```
\\\\\ --> serial code
///// --> parallel code


      +-----+
      |\\\\\|
      |\\\\\|
      +-----+
      |/////|
      |/////|        +-----+
      |/////|        |\\\\\|
      |/////|        +\\\\\|
      |/////|        +-----+
      |/////|        |/////|
---------------------------------------------
       1 processe    N processes
```




Note, the plot can have more stack, to delineate each part of the code how it acts





2 VERSIONS OF THE CODE: 

- V1---> Naive Mat_mul implementation with the 3 nested loops
- V2---> Mat_mul  implementatio used with the dgemm function of blas



And this is only for the CPU part. 

In this course we want to use also the GPU pats. 


new pseudocode:

```
- ALLOCATION OF DISTRIBUTED DATA ON CPU 

- CUDAMEMCPY FROM CPU TO GPU  (A  Cpu --> GPU ) 

- for (c = 0, c < Npes, c++){
    - create_block 
    - ALL_GATHER
    - SET_DEVICE(...)
    - CUDAMEMCPY FROM CPU TO GPU (B_colum CPU --> GPU)
    - Mat_mul_GPU
}
- CUDAMEMCPY FROM GPU TO CPU (C GPU --> CPU)


- PRINT C
```


`CUDA_SET_DEVICE()`  takes a input a number and set the GPU with that number as the current device.
in this way if we have more than one GPU we can select the one we want to use. tipically use in a node is `CUDA_SET_DEVICE(RANK % N_GPUS_IN_NODE)`


In this case the Matrix_multiplication will be done using the `cublasDgemm` function of the CUBLAS library. (the prof said that is usless to implement the matrix multiplication on the GPU by ourself, the library is optimized and we can use it)



Note: in the dgem interface there is a LDC parameters which stands for the leading dimension of the matrix. This is in our case $N$, because the matrix is stored in a linear array.

while the CUBLAS library considered the matrix as a column major matrix, so the leading dimension is the number of rows of the matrix. 

