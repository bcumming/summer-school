CXX = g++
CXXFLAGS = -O3 -mtune=native -ffast-math -g -fopenmp

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .cpp

all: main

.cpp.o: $< $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

main: $(OBJ) main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) *.o main.cpp  -o $@

clean:
	rm -f main
	rm -f *.o
	rm -f *.i

