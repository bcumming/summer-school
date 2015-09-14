
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "omp.h"

double x1[480][500][500];
double x2[3][4][10000000];

int main( int argc, char *argv[] )
{


	
	int ierr, tid, nthreads, i, j, k;
	double time1, time2, x;
	void Init();
	
	// Dont need MPI, just need to use the timers
	ierr = MPI_Init( &argc, &argv );
	

	Init();	
	

	// Start doing OpenMP here
	

	time1 = MPI_Wtime();

#pragma omp parallel for private(i,j,k) collapse(2)
	for(k=0;k<480;k++){
		for(j=0;j<500;j++){
			for(i=0;i<500;i++){
				x1[k][j][i] = exp( x1[k][j][i] );
			}
		}
	}

	time2 = MPI_Wtime();
	printf( "\nTime to complete x1: %e seconds\n", time2 - time1 );
	
	time1 = MPI_Wtime();

#pragma omp parallel for private(i,j,k) collapse(2)
	for(k=0;k<3;k++){
		for(j=0;j<4;j++){
			for(i=0;i<10000000;i++){
				x2[k][j][i] = exp( x2[k][j][i] );
			}
		}
	}
	
	time2 = MPI_Wtime();
	printf( "Time to complete x2: %e seconds\n\n", time2 - time1 );
	



	// Finish up

	time2 = MPI_Wtime();
	ierr = MPI_Finalize();
	

	
	return 0;
}


void Init()
{

	int i, j, k;
	unsigned int iseed;
	int irand;
			
  /* Simple "srand()" seed: just use "time()" */

	iseed = (unsigned int)time(NULL);
    srand(iseed);
	
	
	for(k=0;k<480;k++){
		for(j=0;j<500;j++){
			for(i=0;i<500;i++){
				irand = rand();
				x1[k][j][i] = (double)irand/(double)RAND_MAX;
			}
		}
	}

	
	for(k=0;k<3;k++){
		for(j=0;j<4;j++){
			for(i=0;i<10000000;i++){
				irand = rand();
				x2[k][j][i] = (double)irand/(double)RAND_MAX;
			}
		}
	}


}
