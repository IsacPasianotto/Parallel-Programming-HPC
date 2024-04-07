SHELL=/bin/bash

CC=mpicc

OBJDIR=obj
SRCDIR=src
INCLUDEDIR=include

INCLUDELIBS=-fopenmp

# Uncomment the following line to enable it
# EXTRAFLAGS=-DDEBUG 

CFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -O3 -march=native $(INCLUDELIBS) $(EXTRAFLAGS)
OBJFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -O3 -march=native $(INCLUDELIBS) $(EXTRAFLAGS)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/init.o

main.x: $(OBJS)
	$(CC) $(OBJFLAGS) -o $@ $^
$(OBJDIR)/main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $^
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -f $(OBJDIR)/*.o main.x
