!
! This program calculates pi by integrating the function 4.0/(1+x*x) from 0 to 1.
!

	PROGRAM PIF
	USE MPI
	IMPLICIT NONE
	INTEGER, PARAMETER :: r8 = KIND(1.d0)
	INTEGER, PARAMETER :: i4 = KIND(1)
	
	INTEGER(i4), PARAMETER :: numSteps = 1000000000
	REAL(r8) :: step
	
	INTEGER(i4) :: i, ierr
	REAL(r8) :: x, pi, sum
	REAL(r8) :: time1, time2
	
	!
	! Don't need MPI, just want to use the timers 
	!
	
	CALL MPI_INIT(ierr)
	
	!
	! Initialize some values for the integration
	!
	
	sum = 0.0_r8
	
	step = 1.0_r8/REAL(numSteps,r8)
	
	!
	! Start the timers
	!
	
	time1 = MPI_WTIME()
	
	!
	! Do the integration, OPENMP GOES HERE!!!!
	!
!$omp parallel do private(x) reduction(+:sum)
	DO i = 0, numSteps-1
		x = (REAL(i,r8)+0.50_r8)*step;
		sum = sum + 4.0_r8/(1.0_r8+x*x)
	END DO
!$omp end parallel do
	
	pi = step*sum
	
	!
	! Time to finish up
	!
	
	time2 = MPI_WTIME()
	
	print *
	print *, 'pi = ', pi
	print *, 'Time to calculate pi = ', time2 - time1, ' seconds'
	print *
	
	CALL MPI_FINALIZE()
	
	END PROGRAM PIF
