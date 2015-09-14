
#include <stdio.h>
#include "mpi.h"
#include "omp.h"

const int nn = 41;

int fib( int n );

int main( int argc, char *argv[] )
{


	int ierr, tid, nthreads, i, j, k, result;
	double time1, time2, x;
	
	// Dont need MPI, just need to use the timers
	ierr = MPI_Init( &argc, &argv );
	

	time1 = MPI_Wtime();
	
#pragma omp parallel
#pragma omp single
	result = fib(nn);
	
	time2 = MPI_Wtime();
	printf( "\nResult of fibonacci sequence for n = %d is %d\n", nn, result );
	printf( "Correct answer is 165580141\n\n" );
	printf( "Time is %e seconds \n\n", time2-time1 );	



	// Finish up

	ierr = MPI_Finalize();
	

	
	return 0;
}

int fib( int n){
	int fnum, n1, n2;
	
	if( n < 2 ){
		fnum = n;
		return fnum;
	}
	if( n < 30 ){
		fnum = fib(n-1) + fib(n-2);
		return fnum;
	}
	
#pragma omp task shared(n1)
	n1 = fib(n-1);
	
#pragma omp task shared(n2)
	n2 = fib(n-2);

#pragma omp taskwait

	fnum = n1 + n2;
	
	return fnum;
	
}
