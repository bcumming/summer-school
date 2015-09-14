//
//  This program calculates pi by integrating the function 4.0/(1+x*x) from 0 to 1.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"
#include "omp.h"

void fn1( double *x );
void fn2( double *x );
void fn3( double *x );
void fn4( double *x );
void Init( double *x );
const int n = 50000000;

int main( int argc, char *argv[] )
{

	int i, ierr;
	double a[n], b[n], c[n], d[n], x[n];
	double time1, time2;
	
	
	// Dont need MPI, just need to use the timers

	ierr = MPI_Init( &argc, &argv );

	Init(x);
	
	time1 = MPI_Wtime();
	
	// You could do the following with either parallel for loops, or sections.
	

#pragma omp parallel for	
	for(i=0;i<n;i++){
		a[i] = 0.11111*x[i];
	}
	
#pragma omp parallel for	
	for(i=0;i<n;i++){
		b[i] = -2.0*x[i];
	}
	
#pragma omp parallel for	
	for(i=0;i<n;i++){
		c[i] = 3.14*x[i];
	}
	
#pragma omp parallel for	
	for(i=0;i<n;i++){
		d[i] = 1.5*x[i];
	}

	time2 = MPI_Wtime();
	
	printf( "\nTime for intializations: %e seconds\n", time2 - time1 );
	
	time1 = MPI_Wtime();
	
	//  You should do the following with sections


#pragma omp parallel sections	
{
#pragma omp section
	fn1(a);
	
#pragma omp section
	fn2(b);
	
#pragma omp section
	fn3(c);
	
#pragma omp section
	fn4(d);
}	
	
	// Finish up

	time2 = MPI_Wtime();
	ierr = MPI_Finalize();

	printf( "Time for functions     : %e seconds\n\n", time2 - time1 );
	
	return 0;
}

void Init( double *x )
{

	int i, j, k;
	unsigned int iseed;
	int irand;
			
  /* Simple "srand()" seed: just use "time()" */

	iseed = (unsigned int)time(NULL);
    srand(iseed);

	for(i=0;i<n;i++){
		irand = rand();
		x[i] = (double)irand/(double)RAND_MAX;
	}

}

void fn1( double *x )
{
	int i;
	
	for(i=0;i<n;i++){
		x[i] = exp(x[i]);
	}
}

void fn2( double *x ){
	int i;
	
	for(i=0;i<n;i++){
		x[i] = sin(x[i]);
	}
}

void fn3( double *x ){
	int i;
	
	for(i=0;i<n;i++){
		x[i] = log(x[i]);
	}
}
	
void fn4( double *x ){
	int i;
	
	for(i=0;i<n;i++){
		x[i] = sqrt(x[i]);
	}
}
	
		
