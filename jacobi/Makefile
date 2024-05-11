CC=mpicc
CFLAGS=-O3 -Wall -fopenmp -DSTOPWATCH

# De-comment when you want to use openacc
# CFLAGS += -acc -Minfo=acc
CFLAGS += -acc -Minfo=all

# Comment when you want to plot the final result with gnuplot
CFLAGS += -DFASTOUTPUT

SRCS=$(wildcard *.c)
EXE=$(SRCS:.c=.x)

all: $(EXE)

run: clean all
	./$(EXE) $(dim) $(it) $(r) $(c)

%.x: %.c
	$(CC) $< $(CFLAGS) -o $@

clean:
	@rm -f *~ $(EXE) solution.dat solution*.dat

plot:
	@gnuplot -p plot.plt

plotbin:
	@gnuplot -p plotbin.plt

.PHONY: clean plot all
