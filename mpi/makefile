FTN = ftn

# set compiler-specific flags
ifeq ($(PE_ENV),GNU)
# the gnu compiler requires that we specifically enable OpenMP
	FFLAGS=-O2 -dynamic -fopenmp
endif
ifeq ($(PE_ENV),CRAY)
	FFLAGS= -O2 -dynamic
endif


SOURCES = stats.f90 data.f90 operators.f90 linalg.f90
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .f90

all: main

.f90.o: $<
	$(FTN) $(FFLAGS) $(IFLAGS) -c $< -o $@

main: $(OBJ) main.f90
	$(FTN) $(FFLAGS) *.o main.f90  -o $@
#	$(FTN) $(FFLAGS) *.o main.f90  -o $@ /apps/santis/ddt/4.2.1-35513/lib/64/libdmallocth.a

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f *.mod
	rm -f output.*

