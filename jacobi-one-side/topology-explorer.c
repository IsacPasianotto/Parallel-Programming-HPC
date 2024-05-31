/*---------------------------------------------*
 | file: jacobi.c                              |
 | author: Isac Pasianotto                     |
 | date: 2024-05                               |
 | context: exam of "Parallel programming for  |
 |      HPC". Msc Course in DSSC               |
 | description: Program to explore the topology|
 |      of the machine you are running on.     |
 *---------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <hwloc.h>

void map_numa_cores(int rank, hwloc_topology_t topology, int num_nodes, int num_sockets)
{
  int total_cores = 0;
  for (int i = 0; i < num_nodes; i++)
  {
    hwloc_obj_t numa_node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
    int num_cores_in_node = hwloc_get_nbobjs_inside_cpuset_by_type(topology, numa_node->cpuset, HWLOC_OBJ_CORE);
    printf("Rank %d: NUMA node #%d has %d cores\n", rank, i, num_cores_in_node);

    // List cores in this NUMA node
    for (int j = 0; j < num_cores_in_node; j++)
    {
      hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(topology, numa_node->cpuset, HWLOC_OBJ_CORE, j);
      hwloc_obj_t socket = numa_node->parent;
      int socket_index = socket->os_index;
      int cpu_num = total_cores + j; // Calculate the CPU number dynamically
      printf("  Rank %d:\t CPU %d \tNUMA region %d\t socket %d\n", rank, cpu_num, i, socket_index);
    }
    total_cores += num_cores_in_node; // Update total_cores for the next NUMA node
  }
}

void explore_topology(int rank, int num_sockets)
{
  hwloc_topology_t topology;
  // hwloc_obj_t obj;
  int depth;

  // Initialize hwloc topology
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  // Get the number of NUMA nodes
  depth = hwloc_get_type_depth(topology, HWLOC_OBJ_NUMANODE);
  int num_nodes = hwloc_get_nbobjs_by_depth(topology, depth);
  printf("Rank %d: Number of NUMA nodes: %d\n", rank, num_nodes);

  // Get the number of sockets (packages)
  depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
  printf("Rank %d: Number of sockets: %d\n", rank, num_sockets);

  // Get the total number of cores
  depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
  int num_cores = hwloc_get_nbobjs_by_depth(topology, depth);
  printf("Rank %d: Total number of cores: %d\n", rank, num_cores);

  // Explore the topology of NUMA nodes and cores
  map_numa_cores(rank, topology, num_nodes, num_sockets);

  // Destroy topology object
  hwloc_topology_destroy(topology);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Simulate 1 process per node
  int num_sockets = 2; // Example: Assuming 2 sockets per node
  MPI_Barrier(MPI_COMM_WORLD);
  // print in order of rank
  for (int i = 0; i < world_size; i++)
  {
    if (world_rank == i)
    {
      printf("\n\n============================================\n");
      printf("Rank %d: Exploring topology\n", world_rank);
      explore_topology(world_rank, num_sockets);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }


  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
