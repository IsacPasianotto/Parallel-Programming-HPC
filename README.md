# Parallel-Programming-HPC
This repo was made for the "Parallel Programming for HPC" exam at the University of Trieste that I took during my studies in Data Science and Scientific Computing

***Author***: [Isac Pasianoto](https://github.com/IsacPasianotto/)

- - - 

# TODO: 

- [ ] Add the comments to plot in `mat-mat-mul` exercise
- [ ] Prepare a report for each exercise ($\simeq$ 1 page per exercise)
- 


:> [!WARNING]
> This repository is a work in progress and will be updated as soon as possible.
> At the current moment, I am still following lecture for this course, the exercise will be 
> assigned during this. 



## Table of Contents

- [`mat-mat-mul`](./mat-mat-mul): Implementation of the distributed computation of the *matrix-matrix multiplication* in `C`, done both on CPU and GPU, using `OpenMPI`, `OpenMP`, and `CUDA`.
- [`jacobi`](./jacobi): Porting of a [given code](./jacobi/resources/original_code.c) from CPU to GPU, using `openACC` and distributed with `OpenMPI`.
