#include <iostream>
#include <stdio.h>
#include <cmath>

#include <mpi.h>
#include <omp.h>

#include "data.h"

struct discretization_t cpu::options;

cudaDeviceProp cpu::props;

// fields that hold the solution
__device__ double * __restrict__ gpu::x_old = NULL; // 2d
__device__ double * __restrict__ gpu::bndN = NULL, * __restrict__ gpu::bndE = NULL;
__device__ double * __restrict__ gpu::bndS = NULL, * __restrict__ gpu::bndW = NULL;

__constant__ struct discretization_t gpu::options;


namespace data{

// fields that hold the solution
Field x_new;
Field x_old;

// fields that hold the boundary points
Field bndN;
Field bndE;
Field bndS;
Field bndW;

// buffers used during boundary halo communication
Field buffN;
Field buffE;
Field buffS;
Field buffW;

Discretization options;
SubDomain      domain;

void SubDomain::init(int mpi_rank, int mpi_size, Discretization& discretization)
{
    // determine the number of subdomains in the x and y dimensions
    ndomx = sqrt(double(mpi_size));
    while( mpi_size%ndomx )
        ndomx--;
    ndomy = mpi_size / ndomx;

    // compute this sub-domain index
    // work backwards from: mpi_rank = (domx-1) + (domy-1)*ndomx
    domx = mpi_rank % ndomx + 1;
    domy = (mpi_rank-domx+1) / ndomx + 1;

    nx = discretization.nx / ndomx;
    ny = discretization.ny / ndomy;
    // TODO: the startx and endx values might have to be adjusted by 1
    startx = (domx-1)*nx+1;
    starty = (domy-1)*ny+1;

    // adjust for grid dimensions that do not divided evenly between the
    // sub-domains
    if( domx == ndomx )
        nx = discretization.nx - startx + 1;
    if( domy == ndomy )
        ny = discretization.ny - starty + 1;

    endx = startx + nx -1;
    endy = starty + ny -1;

    // get total number of grid points in this sub-domain
    N = nx*ny;

    neighbour_east  = mpi_rank+1;
    neighbour_west  = mpi_rank-1;
    neighbour_north = mpi_rank+ndomx;
    neighbour_south = mpi_rank-ndomx;

    if (domx == 1) {
        neighbour_west = -1;
    }
    if (domx == ndomx) {
        neighbour_east = -1;
    }
    if (domy == 1) {
        neighbour_south = -1;
    }
    if (domy == ndomy) {
        neighbour_north = -1;
    }

    rank = mpi_rank;
    size = mpi_size;
}

// print domain decomposition information to stdout

void SubDomain::print() {
using std;    
	for(int i=0; i<size; i++) {
        if(rank == i) {
            std::cout << "rank " << rank << "/" << size
                      << " : (" << domx << "," << domy << ")"
                      << " neigh N:S " << neighbour_north << ":" << neighbour_south
                      << " neigh E:W " << neighbour_east << ":" << neighbour_west
                      << " local dims " << nx << " x " << ny
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = omp_get_wtime();
    // add artificial pause so that output doesn't pollute later output
    while(omp_get_wtime()-time < 1e-1);
}

} // namespace data
