FTN = ftn
#FFLAGS += -e Z # use c-style preprocessing
ifeq ($(PE_ENV),GNU)
# the gnu compiler requires that we specifically enable OpenMP
	FFLAGS=-O2 -fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	FFLAGS=-O2 -dynamic
endif

SOURCES = stats.f90 data.f90 operators.f90 linalg.f90
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .f90

all: main

.f90.o: $<
	$(FTN) $(FFLAGS) $(IFLAGS) -c $< -o $@

main: $(OBJ)
	$(FTN) $(FFLAGS) *.o main.f90  -o $@

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f *.mod
	rm -f output.*

