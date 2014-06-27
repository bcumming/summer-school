CXX = gcc
CXXFLAGS = -mtune=native -ffast-math -g -fopenmp
CXXFLAGS = -O0 -fopenmp 

#CXX= cc
#CXXFLAGS=-O3

SOURCES = stats.c data.c operators.c linalg.c linalg_cl.c
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o linalg.o linalg_cl.o

.SUFFIXES: .c

all: main

stats.o: stats.c stats.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c stats.c -o stats.o

data.o: data.c data.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c data.c -o data.o

operators.o: operators.c operators.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c operators.c -o operators.o
	
linalg.o: linalg.c linalg.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c -I /home/aivakhnenko/OpenCL/return/include -I/opt/AMDAPP/include/ linalg.c -o linalg.o -lm -g -L/home/aivakhnenko/OpenCL/return/lib64 -L/opt/clAmdBlas-1.10.321/lib64/ -lclAmdBlas 
	
linalg_cl.o:	linalg_cl.c linalg.h
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c -I/opt/AMDAPP/include/ linalg_cl.c -o linalg_cl.o 
main: $(OBJ) main.c $(HEADERS)
	$(CXX) $(CXXFLAGS) *.o main.c -o main -I /home/aivakhnenko/OpenCL/return/include -I/opt/AMDAPP/include/ -L/home/aivakhnenko/OpenCL/return/lib64 -L/opt/clAmdBlas-1.10.321/lib64/ -l OpenCL -lm -g -lclAmdBlas 

clean:
	rm -f main
	rm -f *.o
	rm -f *.i

