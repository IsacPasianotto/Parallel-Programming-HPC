CC=mpicc
CFLAGS=-O3 -Wall -fopenmp -DSTOPWATCH

# Changing the CFLAGS variable to choose the desired version of the code.
# Uncomment one of the following.
# If none of them is uncommented, the default version is the MPI-Send/Recv version.

CFLAGS += -DONESIDE -DONEWIN -DPUT                            # 1 MPI window, retrieve data with MPI_Put
# CFLAGS += -DONESIDE -DONEWIN                                  # 1 MPI window, retrieve data with MPI_Ge
# CFLAGS += -DONESIDE -DPUT                                     # 2 MPI windows, retrieve data with MPI_Put
# CFLAGS += -DONESIDE                                           # 2 MPI windows, retrieve data with MPI_Get


SRCS=$(wildcard *.c)
EXE=$(SRCS:.c=.x)

all: $(EXE)

run: clean all

	LD_LIBRARY_PATH=$(HWLOC_PATH)/lib:$(LD_LIBRARY_PATH) mpirun -np $(np) ./$(EXE) $(dim) $(it) $(r) $(c)

%.x: %.c
	$(CC) $< $(CFLAGS) -o $@

clean:
	@rm -f *~ $(EXE) solution.dat solution*.dat *.png

plot:
	@gnuplot -p plot.plt

plotbin:
	@gnuplot -p plotbin.plt

.PHONY: clean plot all
