FTN = ftn
FFLAGS = -O2
FFLAGS += -e Z # use c-style preprocessing

SOURCES = stats.f90 data.f90 operators.f90 linalg.f90
OBJ     = stats.o   data.o   operators.o   linalg.o

.SUFFIXES: .f90

all: main

.f90.o: $<
	$(FTN) $(FFLAGS) $(IFLAGS) -c $< -o $@

main: $(OBJ) main.f90
	$(FTN) $(FFLAGS) *.o main.f90  -o $@

clean:
	rm -f main
	rm -f *.o
	rm -f *.i
	rm -f *.mod

