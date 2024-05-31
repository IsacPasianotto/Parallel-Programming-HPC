# Jacobi method for solving Laplace equation

## 0. Table of contents

## 1. Description

This folder contains my solution for the given task: reimplementing the distributed version of the [jacobi exercixe](../jacobi) on cpu,
but this time the `MPI` library must be used with the `one-sided` communication paradigm.

Moreover, the code tries to exploit the topology of the hardware in which is running.

## 2. Before running the jacobi code: topology

The [`topology-explorer`](./topology-explorer.c) will allow you to explore the topology of the hardware in which you are running the code.
It heavily relies on the [`openmpi-hwloc`](https://www.open-mpi.org/projects/hwloc/) library which may be installed on the used cluster. 
In this case you will need to manually compile and link in order to be able to run the code. To do that:

1. Download the source code:
    ```bash
   git clone https://github.com/open-mpi/hwloc.git
    ```
2. Prepare a directory for the build:
    ```bash
    mkdir -p hwloc-build
   ```

3. Configure the build:
   ```bash
   export project_dir=$(pwd)
   cd hwloc
   ./autogen.sh
   ./configure --prefix=$project_dir/hwloc-build --disable-cairo
   ```
4. Compile and install:
    ```bash
    # Assuming you still be in the hwloc directory
    make
    make install 
    ```

To compile the code, you will need to manually link the `hwloc` library.
Moreover, in order to let the code run properly, you will need to set the `LD_LIBRARY_PATH` environment variable to the path where the `hwloc` library is installed, as showed in the example [`sbatcher-topology.sh`](./sbatcher-topology.sh) script.

>*Remark:* Depending on the system you are running the code, some function may not work properly.
> I have assessed the correctness of the code on THIN and EPYC partition of the [Orfeo cluster](https://www.areasciencepark.it/piattaforme-tecnologiche/data-center-orfeo/). 
> However, in the case of the DCGP partition of the Leonardo cluster, the function which retrieves in which socket the core is returns the -1 error value.

### 2.1 Results of the topology-explorer

After running the code (the output is recorded in the file [`topology.log`](./topology.log)), I have obtained that nodes in the DCGP partition have the following topology:

- The total number of cores is 112
- The node has 2 cpu sockets
- each cpu has in total 56 cores
- each cpu has 4 NUMA regions 
- each NUMA region has 14 cores

Note: All these conclusion are based on the reasonable hypothesis that the two sockets are identical. 

## 3. Modification to the original code

TBD...