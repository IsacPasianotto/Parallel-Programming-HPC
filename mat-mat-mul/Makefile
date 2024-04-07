SHELL=/bin/bash

CC=mpicc

OBJDIR=obj
SRCDIR=src
INCLUDEDIR=include

INCLUDELIBS=-fopenmp

# Uncomment the following line to enable it
EXTRAFLAGS+=-DDEBUG
# EXTRAFLAGS+=-O3  # kept separate because causes issues with printing order in debug mode

CFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -march=native $(INCLUDELIBS) $(EXTRAFLAGS)
OBJFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -march=native $(INCLUDELIBS) $(EXTRAFLAGS)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/init.o $(OBJDIR)/debug.o

main.x: $(OBJS)
	$(CC) $(OBJFLAGS) -o $@ $^
$(OBJDIR)/main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $^
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -f $(OBJDIR)/*.o main.x
