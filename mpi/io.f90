module io

! dependencies
use mpi
use data,   only: subdomainT, discretizationT, options, domain

implicit none

contains

subroutine write_parallel(filename, u)
    ! arguments
    real (kind=8), intent(in) :: u(options%nx,options%ny)
    character(len=*)          :: filename

    ! variables
    integer(kind=mpi_offset_kind) :: disp
    integer :: ierr, filehandle, filetype, N
    integer :: dimuids(2), ustart(2), ucount(2)

    ! initial displacement is zero
    disp = 0

    ! open file handle
    call mpi_file_open(                         &
        MPI_COMM_WORLD, Filename,               &
        ior(MPI_MODE_CREATE, MPI_MODE_WRONLY),  &
        MPI_INFO_NULL, filehandle, ierr)

    ! calculate field dimensions
    ustart(1) = domain%startx-1
    ustart(2) = domain%starty-1

    ucount(1) = options%nx
    ucount(2) = options%ny
    N = ucount(1) * ucount(2)

    dimuids(1) = options%global_nx
    dimuids(2) = options%global_ny

    ! write header
    !if (domain%rank == 0) then
        !call mpi_file_write(filehandle, dimuids, 1, MPI_INTEGER, MPI_STATUS_IGNORE, ierr)
    !endif
    !disp = sizeof(dimuids)

    ! create a subarray representing the local block
    call MPI_Type_create_subarray(2, dimuids, ucount, ustart, &
                                       MPI_ORDER_FORTRAN, MPI_DOUBLE, filetype, ierr)
    call MPI_Type_commit(filetype, ierr)

    call MPI_File_set_view(filehandle, disp, MPI_DOUBLE, &
                           filetype, "native", MPI_INFO_NULL, ierr)

    call MPI_File_write_all(filehandle, u, N, MPI_DOUBLE, MPI_STATUS_IGNORE, ierr)
    call MPI_Type_free(filetype, ierr)
    call MPI_File_close(filehandle, ierr)
end subroutine write_parallel

end module io
