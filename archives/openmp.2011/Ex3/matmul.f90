      PROGRAM MATMULT
                USE MPI
      INTEGER  NRA, NCA, NCB, TID, NTHREADS, I, J, K, CHUNK, &
              OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM, ierr
!     number of rows in matrix A 
      PARAMETER (NRA=1024)
!     number of columns in matrix A
      PARAMETER (NCA=512)
!     number of columns in matrix B
      PARAMETER (NCB=256)

      REAL*8 A(NRA,NCA), B(NCA,NCB), C(NRA,NCB)
          REAL(8) time1,  time2
          
          CALL MPI_INIT(ierr)

          time1 = MPI_WTIME()

!     Spawn a parallel region explicitly scoping all variables

!$omp parallel private(tid) shared(nthreads)

      TID = OMP_GET_THREAD_NUM()
      NTHREADS = OMP_GET_NUM_THREADS()

!         For this bit, consider a single thread
!$omp single
        PRINT *, 'Starting matrix multiple example with', NTHREADS, &
                'threads'
!$omp end single
          
          ! do some Fortran sections

!$omp sections
!$omp section
            CALL RANDOM_NUMBER(A)
!$omp section
          CALL RANDOM_NUMBER(B)
!$omp section
          C(:,:) = 0.0
!$omp end sections

!     Do matrix multiply sharing iterations on outer loop
!     Display who does which iterations for demonstration purposes

      PRINT *, 'Thread', TID, 'starting matrix-matrix multiply...'

!$omp do
      DO  J=1, NCB
        DO  I=1, NRA
          DO  K=1, NCA
            C(I,J) = C(I,J) + A(I,K) * B(K,J)
                  END DO
                END DO
          END DO
!$omp end do
!$omp end parallel

          time2 = MPI_WTIME()
          
          PRINT *
          PRINT *, 'Time for ', nthreads, ' threads : ', time2-time1
          PRINT *


      END
