# Jacobi method for solving Laplace equation

## 0. Table of contents

***TODO***


## 1. Description

This folder contains my solution for the given task: taking an already existing code, in this case the [`original_code.c`](./resources/original_code.c) and porting it to the GPU.
To do so I've used the `openACC` library. Moreover, the code has also been modified to implement also `MPI`, in order to distribute the computation among multiple nodes and OpenMP to parallelize the computation among different cores in the same node. 

## 2. Modification to the original code

### 2.1. OpenMP

The OpenMP directive influences the initialization of the matrix and the computation of the values of the matrix for the following iteration: 

*Remark:* the parallelization of the main loop is used only if `openACC` is not used (when the code is runned on CPU and not on GPU)

### 2.2. OpenMPI

The matrix is divided in blocks, and each blocks is assigned to a process. If the division is not perfect the first processes will take care of the remaining rows.
This is done for both the matrix that is storing the current values of the matrix and the matrix that is storing the values of the matrix for the following iteration (at the end of the iteration the two pointers are swapped).

***TODO: inserire immagine divisioni matrici***

To be more precise, in order to carry on computation, every process need to communicate with the others. Hence the `local_size` values of the previous figure is not the whole memory allocation each process need to have available. In fact each (with except of the first and the last) processes will store two additional more rows, one on the top and one at the bottom. 

***TODO: inserire immagine communicazione***


When the computation is concluded we need to store the results somehow. I have implemented two different ways to do so:

- A faster way, based on `MPI-IO`, in which each proces write its own block in the file.
- A slower way, in which the master process collects all the blocks and then writes them in the file using the already existing `save_gnuplot` function.


> **Note**: The second implementation should be avoided. I used that because `MPI-IO` writes binary files and I discovered only later that `gnuplot` was able to interpreter them properly modify the given [`plot.plt`](./plot.plt) file as show in [`plotbin.plt`](./plotbin.plt). The difference in terms of time is huge, even one or two orders of magnitude.

### 2.3. openACC
