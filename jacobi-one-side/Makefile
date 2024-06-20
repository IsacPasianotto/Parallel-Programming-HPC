CC=mpicc
CFLAGS=-O3 -Wall -fopenmp -lm -lhwloc
# CFLAGS += ...
FLAGS += -DONESIDE

# Comment when you want to plot the final result with gnuplot
CFLAGS += -DFASTOUTPUT

SRCS=$(wildcard *.c)
EXE=$(SRCS:.c=.x)

all: $(EXE)

run: clean all
	LD_LIBRARY_PATH=$(HWLOC_PATH)/lib:$(LD_LIBRARY_PATH) mpirun -np $(np) ./$(EXE) $(dim) $(it) $(r) $(c)

%.x: %.c
	$(CC) $< $(CFLAGS) -o $@

clean:
	@rm -f *~ $(EXE) solution.dat solution*.dat

plot:
	@gnuplot -p plot.plt

plotbin:
	@gnuplot -p plotbin.plt

.PHONY: clean plot all
