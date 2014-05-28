FTN = ftn

FFLAGS = -rmd -hacc -O3

SOURCES = stats.f90 data.f90 operators.f90 linalg.f90
OBJ     = stats.o   data.o   operators.o   linalg.o
OBJ_MPI = stats.o   data.o   operators_mpi.o   linalg.o

.SUFFIXES: .f90

all: main main_mpi

stats.o: stats.f90
	$(FTN) $(FFLAGS) $(IFLAGS) -c stats.f90 -o stats.o

data.o: data.f90
	$(FTN) $(FFLAGS) $(IFLAGS) -c data.f90 -o data.o

operators.o: operators.f90
	$(FTN) $(FFLAGS) $(IFLAGS) -c operators.f90 -o operators.o

operators_mpi.o: operators_mpi.f90
	$(FTN) $(FFLAGS) $(IFLAGS) -c operators_mpi.f90 -o operators_mpi.o

linalg.o: linalg.f90
	$(FTN) $(FFLAGS) $(IFLAGS) -c linalg.f90 -o linalg.o

main: $(OBJ) main.f90
	$(FTN) $(FFLAGS) $(OBJ) main.f90  -o main

main_mpi: $(OBJ_MPI) main.f90
	$(FTN) $(FFLAGS) $(OBJ_MPI) main.f90  -o main_mpi

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f *.mod
	rm -f *.lst
	rm -f *.opt
	rm -f *.cg
	rm -f *.ptx
	rm -f *.cub
	rm -f output.*

