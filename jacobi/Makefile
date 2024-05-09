# inputs for the executables

dim = 10
it  = 100
r = 3
c = 3

CC=mpicc
# enable the openACC compiler
# CFLAGS=-O3 -fopenacc -lm
# CFLAGS=-O3 -acc -Minfo=all -DDEBUG_OPENACC
CFLAGS=-O3 -DSTOPWATCH# -acc -Minfo=all -DDEBUG_OPENACC -DFASTOUTPUT

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
