# Parallel-Programming-HPC
This repo was made for the "Parallel Programming for HPC" exam at the University of Trieste that I took during my studies in Data Science and Scientific Computing

***Author***: [Isac Pasianoto](https://github.com/IsacPasianotto/)
***Course***: Parallel Programming for HPC, University of Trieste, Italy
***Course repository***: [Parallel Programming for HPC / Advanced High Performance Computing](https://github.com/Foundations-of-HPC/Advanced-High-Performance-Computing-2023/)

## Table of Contents

- [`mat-mat-mul`](./mat-mat-mul): Implementation of the distributed computation of the *matrix-matrix multiplication* in `C`, done both on CPU and GPU, using `OpenMPI`, `OpenMP`, and `CUDA`.
- [`jacobi`](./jacobi): Porting of a [given code](./jacobi/resources/original_code.c) from CPU to GPU, using `openACC` and distributed with `OpenMPI`.
- [`jacobi-one-side`](./jacobi-one-side): Re-implementation of the solution proposed in the `jacobi` section, but using the `MPI Remote Memory Access (RMA)` approach.

For all the sections, the `README.md` file contains a detailed description of the work done, the requirements, the implementation, and the results obtained, moreover a `report.pdf` file is provided with a summary of the work done and the results obtained.

## Remark
The results presented in the [`mat-mat-mul`](./mat-mat-mul) and [`jacobi`](./jacobi) sections are **biased**.\
In fact, it turned out  that running the code with `mpirun -np` leaded to a single thread execution.\
Digging into the problem, It turned out that the same issue does not occur when running the code with `srun -N` command.

I figured out it in time to present the results of the `jacobi-one-side` executed in the proper way (I have also kept a first wrong execution in the results, to show the difference in [`one_thread`](./jacobi-one-side/plots/one_thread) folder). 
I considered to re-measure the results of the other sections, but at the moment I am writing this (2024-06-28) the availability of the resources was almost zero and the deadline is too close.

```
[ipasiano@login07 ~]$ saldo --b --dcgp
-----------------------------------------------------------------------------------------------------------------------------------------
account                start         end         total        localCluster   totConsumed     totConsumed     monthTotal     monthConsumed
                                             (local h)   Consumed(local h)     (local h)               %      (local h)         (local h)
-----------------------------------------------------------------------------------------------------------------------------------------
ICT24_DSSC_CPU      20240314    20240831         39000               38734         38734            99.3          6882              20624
```

However, for both the `mat-mat-mul` and `jacobi` sections, a not complete scaling analysis - actually just one run for the 1 node case - is presented, in order to have a rough idea of the performance of the code.
