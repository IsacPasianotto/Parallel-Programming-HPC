## Standard preamble 

SHELL=/bin/bash
CC=mpicc
NVCC=nvcc
mainfile=main.c

OBJDIR=obj
SRCDIR=src
INCLUDEDIR=include
INCLUDELIBS=-fopenmp

##  Standard compiler flags

CFLAGS=-I$(INCLUDEDIR) -march=native $(INCLUDELIBS) $(EXTRAFLAGS)
OBJFLAGS=-I$(INCLUDEDIR) -march=native $(INCLUDELIBS) $(EXTRAFLAGS)

##  Debug flags

# Uncomment the following line to enable it
# EXTRAFLAGS+=-DDEBUG
# EXTRAFLAGS+=-DSMALL
# EXTRAFLAGS+=-DSMALL_INTS
# EXTRAFLAGS+=-DDEBUG_INIT
# EXTRAFLAGS+=-DDEBUG_COL_BLOCK
# EXTRAFLAGS+=-DDEBUG_COL_GATHER
# EXTRAFLAGS+=-DDEBUG_PROD
# EXTRAFLAGS+=-DNOSTOPWATCH

EXTRAFLAGS+=-O3  # kept separate because causes issues with printing order in debug mode


## Implementation specific things

BLASFLAGS=-DOPENBLAS -I/usr/include/openblas -L/usr/lib -lopenblas
CUDAFLAGS=-I$(INCLUDEDIR) -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib -lmpi -lcudart -lcublas -DCUDA

## Objects definitions

OBJS=$(OBJDIR)/main.o \
     $(OBJDIR)/init.o \
     $(OBJDIR)/debug.o \
     $(OBJDIR)/column_gathering.o \
     $(OBJDIR)/product.o \
     $(OBJDIR)/stopwatch.o

##  Compilation 

blas: CFLAGS += $(BLASFLAGS)
blas: $(OBJS)
	$(CC) $(OBJFLAGS) -o main.x $^ $(BLASFLAGS)

cuda: CC := $(NVCC)
cuda: CFLAGS := $(CUDAFLAGS)
cuda: OBJFLAGS := $(CUDAFLAGS)
cuda: $(OBJS)
	$(NVCC) $(CUDAFLAGS) -o main.x $^ $(CUDAFLAGS)


main.x: $(OBJS)
	$(CC) $(OBJFLAGS) -o $@ $^

$(OBJDIR)/main.o: $(mainfile)
	$(CC) $(CFLAGS) -c -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^




# needed?
.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o main.x

