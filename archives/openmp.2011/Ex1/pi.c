//
//  This program calculates pi by integrating the function 4.0/(1+x*x) from 0 to 1.
//

#include <stdio.h>
#include "mpi.h"


int main( int argc, char *argv[] )
{

	int i, ierr;
        int numSteps = 1000000000;
	double sum = 0.0;
	double step;
	double time1, time2;
	double x, pi;
	
	
	// Dont need MPI, just need to use the timers

	ierr = MPI_Init( &argc, &argv );
	time1 = MPI_Wtime();
	
	// Initialize some values for integration
	
	sum = 0.0;
	step = 1.0/(double)numSteps;
	
	// Integrate using a for loop, OpenMP goes here!!!!!!!!

#pragma omp parallel for private(x) reduction(+:sum)
	for( i=0; i<numSteps; i++ )
	{
		x = ((double)i + 0.50)*step;
		sum = sum + 4.0/(1.0 + x*x );
	}
	
	pi = step * sum;
	
	// Finish up

	time2 = MPI_Wtime();
	ierr = MPI_Finalize();
	
	printf( "\n" );
	printf( "pi = %e\n", pi );
	printf( "Time to calculate pi = %e seconds\n\n" , time2-time1 );

	
	return 0;
}
