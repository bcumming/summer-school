      PROGRAM WORK
        USE MPI
        IMPLICIT NONE
        INTEGER N, I, IERR, omp_get_thread_num
        PARAMETER (N=10000000)
        REAL(8) A(N), B(N), C(N), D(N), X(N)
        REAL(8) TIME1, TIME2
    
        CALL MPI_INIT(ierr)

!     Some initializations (note these are Fortran array operations!)
!     Consider how you might do this with work sharing (parallel workshare and sections)

        CALL RANDOM_NUMBER(X)

        time1 = MPI_WTIME()
          
!!! worksharing section !!!

!$omp parallel workshare
          A = X
          B = -2.0*X
          C = 3.14*X
          D = 1.5*X
!$omp end parallel workshare

          time2 = MPI_WTIME()
          
          print *, 'Time to do intializations : ', time2-time1
          
          !!! Sections section!
          
          time1 = MPI_WTIME()
          
!$omp parallel sections
!$omp section
          CALL FN1(A)
!$omp section
          CALL FN2(B)
!$omp section
          CALL FN3(C)
!$omp section
          CALL FN4(D)
!$omp end parallel sections

          time2 = MPI_WTIME()
        
          PRINT *, 'Time for sections : ', time2-time1

          CALL MPI_FINALIZE()
          
          
          CONTAINS
          
          SUBROUTINE FN1( X )
                  REAL(8) :: X(:)
                X = EXP(X)
          END SUBROUTINE FN1

          SUBROUTINE FN2( X )
                  REAL(8) :: X(:)
                X = SIN(X)
          END SUBROUTINE FN2

          SUBROUTINE FN3( X )
                  REAL(8) :: X(:)
                X = LOG(X)
          END SUBROUTINE FN3
          
          SUBROUTINE FN4( X )
                  REAL(8) :: X(:)
                X = SQRT(X)
          END SUBROUTINE FN4
          
          END PROGRAM WORK
