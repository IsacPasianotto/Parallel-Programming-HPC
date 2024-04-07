CC=gcc

OBJDIR=obj
SRCDIR=src
INCLUDEDIR=include

INCLUDELIBS=-fopenmp

# To pass arguments like DEBUG, etc to the compiler
INPUT_DEFS := $(filter-out $@,$(MAKECMDGOALS))
EXTRA_FLAGS += $(foreach flag,$(ADDITIONAL_FLAGS),-D$(flag))

CFLAGS=-I$(INCLUDEDIR) -Wall -Wextra -O3 -march=native $(INCLUDELIBS) $(EXTRA_FLAGS)
OBJFLAGS=-Wall -Wextra -O3 -march=native $(INCLUDELIBS) $(EXTRA_FLAGS)


OBJS=$(OBJDIR)/main.o

main.x: $(OBJS)
	$(CC) $(OBJFLAGS) -o $@ $^
$(OBJDIR)/main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $^
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -f $(OBJDIR)/*.o main.x
