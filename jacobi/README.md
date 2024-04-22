# Jacobi method for solving Laplace equation

## Table of contents

***TODO***


## Description

This folder contains my solution for the given task: taking an already existing code, in this case the [`original_code.c`](./resources/original_code.c) and porting it to the GPU.
To do so I've used the `openACC` library. Moreover, the code has also been modified to implement also `MPI`, in order to distribute the computation among multiple nodes.

## Modification

The modification I have made to achieve the desired result are the following:

### Makefile

I modified the `Makefile` to include the compilation of the `openACC` code. Moreover, the compiler was changed from `gcc` to `mpicc` to include the `MPI` library.

