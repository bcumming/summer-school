
!
! Parallelize the following loops both with and without the collapse(2) directive
! Run for different thread counts and note performance
!




	PROGRAM COLL
	USE MPI
	IMPLICIT NONE
	
	INTEGER :: i,j,k, ierr
	
	REAL(8) :: time1, time2, x, b, r, sum0
	
	REAL(8) :: X1(500,500,480)
	REAL(8) :: X2(10000000,4,3)
	
	CALL RANDOM_NUMBER(x1)
	CALL RANDOM_NUMBER(x2)
	
		
	time1 = MPI_WTIME()
	
	! loop construct 1
	
!$omp parallel do collapse(2)
	DO i = 1, 480
		DO j = 1, 500
			DO k = 1, 500
				X1(k,j,i) = EXP(X1(k,j,i))
			END DO
		END DO
	END DO
!$omp end parallel do

	x2(1,1,1) = x1(1,1,1)	
	
	time2 = MPI_WTIME()
	
	PRINT *
	PRINT *, 'Time to complete X1 ',  time2 - time1, ' seconds'
	
	time1 = MPI_WTIME()
	
	! loop construct 2

!$omp parallel do collapse(2)
	DO i = 1, 3
		DO j = 1, 4
			DO k = 1, 10000000
				X2(k,j,i) = EXP(X2(k,j,i))
			END DO
		END DO
	END DO
!$omp end parallel do
	
	
	time2 = MPI_WTIME()

	x1(1,1,1) = x2(1,1,1)
	
	PRINT *, 'Time to complete X2 ',  time2 - time1, ' seconds'
	PRINT *
	
	
	
	END PROGRAM COLL
