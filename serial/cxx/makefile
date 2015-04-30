# by default select gcc
CXX=g++
CXXFLAGS=-O3 -fopenmp

# if on a cray system, set flags according to the compilation enviroment
ifeq ($(PE_ENV),GNU)
	CXX=CC
	CXXFLAGS=-O3 -fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	CXX=CC
	CXXFLAGS=-O3
	CXXFLAGS+=-h list=a
endif
ifeq ($(PE_ENV),INTEL)
	CXX=CC
	CXXFLAGS=-O3 -openmp
endif
ifeq ($(PE_ENV),PGI)
	CXX=CC
	CXXFLAGS=-O3 -mp
endif

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .cpp

all: main

stats.o: stats.cpp stats.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c stats.cpp -o stats.o

data.o: data.cpp data.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c data.cpp -o data.o

operators.o: operators.cpp operators.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c operators.cpp -o operators.o

linalg.o: linalg.cpp linalg.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c linalg.cpp -o linalg.o

main: $(OBJ) main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) *.o main.cpp -o main

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f *.lst
	rm -f output.*

