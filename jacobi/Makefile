CC=mpicc
CFLAGS=-O3 -DSTOPWATCH -DFASTOUTPUT

# Decomment when you want to use openacc
CFLAGS += -acc -Minfo=acc

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

.PHONY: clean plot all
