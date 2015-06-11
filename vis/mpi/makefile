CXX=CC

CXXFLAGS=-O3 -fopenmp
CXXFLAGS=-O2  -fopenmp -D_VISIT_ -I${VISITHOME}/current/linux-x86_64/libsim/V2/include
LINKFLAGS=-L${VISITHOME}/current/linux-x86_64/libsim/V2/lib -rdynamic  -lsimV2 -ldl -lm -lstdc++ -Wl,-rpath,${VISITHOME}/current/linux-x86_64/libsim/V2/lib


ifeq ($(PE_ENV),GNU)
    # the gnu compiler requires that we specifically enable OpenMP
	#CXXFLAGS=-O0 -g -fopenmp -DDEBUG
CXXFLAGS=-dynamic -O2  -fopenmp -D_VISIT_ -I${VISITHOME}/current/linux-x86_64/libsim/V2/include
LINKFLAGS=-L${VISITHOME}/current/linux-x86_64/libsim/V2/lib -rdynamic  -lsimV2 -ldl -lm -lstdc++ -Wl,-rpath,${VISITHOME}/current/linux-x86_64/libsim/V2/lib
endif
ifeq ($(PE_ENV),CRAY)
	CXXFLAGS=-O3
endif

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp visit.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h
OBJ     = stats.o   data.o   operators.o   linalg.o visit.o

.SUFFIXES: .cpp

all: main

stats.o: stats.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c stats.cpp -o stats.o

data.o: data.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c data.cpp -o data.o

operators.o: operators.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c operators.cpp -o operators.o

linalg.o: linalg.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c linalg.cpp -o linalg.o

main: $(OBJ) main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) *.o main.cpp -o main  ${LINKFLAGS}

clean:
	rm -f main
	rm -f *.o
	rm -f *.i

