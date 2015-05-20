
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

	int main( void )
	{
	
		long int i, j;
		int k,thrID,n, ierr;
		double x, y, r2, time0, time1, time2;
		double a[10000][10000],b[10000][10000];
		FILE *fp;
		
		n=0;
		ierr = MPI_Init_thread( &n, NULL, MPI_THREAD_SERIALIZED, &k );
	
		/*
		 Unthreaded version, do not touch!
		*/

		time1 = MPI_Wtime();
		
		for(i=0;i<10000;i++){
				y = (double)(i-5000)/5000.0;
			for(j=0;j<10000;j++){
				x = (double)(j-5000)/5000.0;
				r2 = x*x + y*y;
				a[i][j] = exp(-r2);
			}
		}
		
		fp = fopen( "overlap.out", "w" );
		n = 10000*10000;
		for(i=0;i<3;i++)
			fwrite( a, n, sizeof(double), fp );
		fclose(fp);
		
			
		time2 = MPI_Wtime();		
		printf( "Time to complete, unmodified code = %f\n", time2 - time1 );
	

		/*
		 Threaded version, start editing here!
		*/

		time1 = MPI_Wtime();
#pragma omp parallel private(thrID)
{
		thrID = omp_get_thread_num();
		if( thrID > 0 ){
#pragma omp parallel for private(x,y,r2)  num_threads(11) 
			for(i=0;i<10000;i++){
				y = (double)(i-5000)/5000.0;
				for(j=0;j<10000;j++){
					x = (double)(j-5000)/5000.0;
					r2 = x*x + y*y;
					a[i][j] = exp(-r2);
				}
			}
		}
		else
		{
		fp = fopen( "overlap.out", "w" );
		n = 10000*10000;
		for(i=0;i<3;i++)
			fwrite( b, n, sizeof(double), fp );
		fclose(fp);
		}
}
					
		time2 = MPI_Wtime();		
		printf( "Time to complete, modified code = %f\n", time2 - time1 );
	
		ierr = MPI_Finalize();
	
	}


