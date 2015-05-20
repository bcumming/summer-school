
#include <stdio.h>
#include "mpi.h"
#include "omp.h"

int main( int argc, char *argv[] )
{

	int nra = 1024;
	int nca = 1024;
	int ncb = 256;
	
	double a[nra][nca];
	double b[nca][ncb];
	double c[nra][ncb];
	
	int ierr, tid, nthreads, i, j, k;
	double time1, time2;
	
	
	// Dont need MPI, just need to use the timers

	ierr = MPI_Init( &argc, &argv );
	time1 = MPI_Wtime();

	// Start doing OpenMP here
	
#pragma omp parallel shared(nthreads,a,b,c) private(tid,nthreads,i,j,k)
{

	// For the next bit, consider printing out only one thread.

	tid = omp_get_thread_num();
	nthreads = omp_get_num_threads();

#pragma omp single
{
	printf( "Starting matrix multiply example with %d threads\n", nthreads );	
}

	// initialize matrices A and C

	for(i=0; i<nra; i++ ){
		for(j=0; j<nca; j++ ){
			a[i][j] = (double)(i*j)/(double)(nra*nca);
		}
		for(j=0; j<ncb; j++ ){
			c[i][j] = 0.0;
		}
	}
	
	// Initialize matrix B
	
#pragma omp for
	for(i=0; i<nca; i++ ){
		for(j=0; j<ncb; j++ ){
			b[i][j] = (double)(i*i)/(double)(nca*nca);
		}
	}
	
	printf( "Thread %d starting matrix-matrix multiply\n", tid );

#pragma omp for	
	for(i=0; i<nra; i++ ){
		for(j=0; j<ncb; j++ ){
			for(k=0; k<nca; k++ ){
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}


}	
	// Finish up

	time2 = MPI_Wtime();
	ierr = MPI_Finalize();
	
	printf( "\nTime for %d threads : %e seconds \n\n" , nthreads, time2-time1 );

	
	return 0;
}
