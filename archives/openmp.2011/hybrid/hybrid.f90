        PROGRAM hybrid
                use mpi
                implicit none
            integer, parameter :: r8 = selected_int_kind(15)
                REAL(r8) :: a(10000,10000), b(10000,10000)
                INTEGER*8 :: i,j, k
                INTEGER :: thrID, n, index, offset, omp_get_thread_num, omp_get_num_threads, ierr, num_threads, n1, n2
                REAL(r8) :: x, y, r2, time0, time1, time2, ovr, dt
                
                CALL MPI_INIT_THREAD( MPI_THREAD_SERIALIZED, n, ierr )
                
                !
                ! Unthreaded version, do not touch!
                !

                ! compute
                time1 = MPI_WTIME()                
                DO j = 1, 10000
                        y = real(j-5000,r8)/5000.0_r8
                        DO i = 1, 10000
                                x = real(i-5000,r8)/5000.0_r8
                                r2 = x*x + y*y
                                a(i,j) = EXP(-r2)
                        END DO
                END DO

                ! i/o
                OPEN(10,file='overlap.out',form='unformatted')
                DO i = 1, 3
                WRITE(10) b
                END DO
                CLOSE(10)         
                        
                time2 = MPI_WTIME()
                PRINT *,'Time to complete, unmodified code = ', time2 - time1 

                !
                ! Threaded version, start editing here!
                !
                
                time1 = MPI_WTIME()                

                !compute
                DO j = 1, 10000
                        y = real(j-5000,r8)/5000.0_r8
                        DO i = 1, 10000
                                x = real(i-5000,r8)/5000.0_r8
                                r2 = x*x + y*y
                                a(i,j) = EXP(-r2)
                        END DO
                END DO

                !i/o
                OPEN(10,file='overlap.out',form='unformatted')
                DO i = 1, 3
                WRITE(10) b
                END DO
                CLOSE(10)         


                time2 = MPI_WTIME()
                PRINT *,'Time to complete, modified code = ', time2 - time1 
                
                
                CALL MPI_FINALIZE( ierr )
        
        END PROGRAM hybrid
