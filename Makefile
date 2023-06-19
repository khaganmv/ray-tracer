.PHONY: run clean valgrind nvprof

INCLUDEDIR = include
SRCDIR = src/*.cu
BIN = out/main

all: $(BIN)

$(BIN): $(SRCDIR)
	nvcc $^ -o $@ -I$(INCLUDEDIR) -O3

run: all
	./$(BIN)

clean:
	rm -f out/main

valgrind: all
	valgrind --leak-check=full ./$(BIN)

nvprof:
	nvprof ./$(BIN)
