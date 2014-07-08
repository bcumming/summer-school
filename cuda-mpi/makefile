PE_ENV=GNU
ifeq ($(PE_ENV),GNU)
	MPICXX=mpicxx
	CXXFLAGS=-g -O3 -fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	MPICXX=CC
	CXXFLAGS=-O3
endif

#
# Locate folder with mpi.h and add it to include paths
# (needed for non-MPI compiler that uses MPI-aware headers, e.g. CUDA).
#
CINC += -I$(shell dirname $(shell echo "\#include <mpi.h>" | $(MPICC) -xc - -M | grep -m 1 -o -e "[^ ]*mpi\.h"))

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o    main.o

.SUFFIXES: .cpp

all: main

stats.o: stats.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

data.o: data.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(IFLAGS) -c $< -o $@ -I/opt/cuda/include/

operators.o: operators.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

linalg.o: linalg.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

init_x_new.o: init_x_new.cu $(HEADERS)
	nvcc $(CXXFLAGS) $(IFLAGS) -c $< -o $@ 

#mpicxx -g -O3 -fopenmp  -c data.cpp -o data.o -I/opt/cuda/include/


main: $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f main
	rm -f *.o
	rm -f *.i

