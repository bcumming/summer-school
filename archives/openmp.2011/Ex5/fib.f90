program main
	use mpi
  implicit none
  interface 
     recursive function fib(n)
       integer, intent(in) :: n 
       integer :: fib
     end function fib
  end interface
  integer :: result, ierr
  integer, parameter :: nn = 41
  real(8) :: t1, t2
  
	call MPI_INIT(ierr)
	
	t1 = MPI_WTIME()
	
	!
	! start here. how many threads should call fib?
	!
!$omp parallel 
!$omp single
	result = fib(nn)
!$omp end single
!$omp end parallel

	t2 = MPI_WTIME()

  print *
  print *, 'Result of fibonacci sequence for n = ', nn, ' is ', result
  print *, 'Correct answer is 165580141'
  print *
  print *, 'Time is ', t2-t1
  print *
  
  CALL MPI_FINALIZE(ierr)
  
end program main

recursive function fib (n)  result (fnum) 
  integer, intent(in)  :: n
  integer :: fnum, n1, n2
  if (n<2)then
  	fnum = n
	return
  end if
  if( n < 30 )then
  	fnum = fib(n-1) + fib(n-2)
    return 
  end if

!$omp task shared(n1)
  	 n1 = fib(n-1)
!$omp end task
	 
!$omp task shared(n2)
	 n2 = fib(n-2)
!$omp end task
!$omp taskwait
     fnum = n1 + n2

end function fib
