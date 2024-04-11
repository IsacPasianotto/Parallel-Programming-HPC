SHELL=/bin/bash

CC=mpicc

mainfile=main.c

OBJDIR=obj
SRCDIR=src
INCLUDEDIR=include

INCLUDELIBS=-fopenmp

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

CFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -march=native $(INCLUDELIBS) $(EXTRAFLAGS)
OBJFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -march=native $(INCLUDELIBS) $(EXTRAFLAGS)

BLASFLAGS=-DOPENBLAS -I/usr/include/openblas -L/usr/lib -lopenblas

OBJS=$(OBJDIR)/main.o \
     $(OBJDIR)/init.o \
     $(OBJDIR)/debug.o \
     $(OBJDIR)/column_gathering.o \
     $(OBJDIR)/product.o \
     $(OBJDIR)/stopwatch.o

main.x: $(OBJS)
	$(CC) $(OBJFLAGS) -o $@ $^
$(OBJDIR)/main.o: $(mainfile)
	$(CC) $(CFLAGS) -c -o $@ $^
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

blas: CFLAGS += $(BLASFLAGS)
blas: $(OBJS)
	$(CC) $(OBJFLAGS) -o main.x $^ $(BLASFLAGS)

clean:
	rm -f $(OBJDIR)/*.o main.x
