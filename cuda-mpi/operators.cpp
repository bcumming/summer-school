//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include <mpi.h>

#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

void diffusion(const data::Field &U, data::Field &S)
{
    using data::options;
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = domain.nx;
    int ny = domain.ny;
    int iend  = nx - 1;
    int jend  = ny - 1;

    MPI_Status statuses[8];
    MPI_Request requests[8];
    int num_requests = 0;

    if(domain.neighbour_north>=0) {
        // set tag to be the sender's rank
        // post receive
        MPI_Irecv(&bndN[0], nx, MPI_DOUBLE, domain.neighbour_north, domain.neighbour_north,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;

        // pack north buffer
        for(int i=0; i<nx; i++)
            buffN[i] = U(i,ny-1);

        // post send
        MPI_Isend(&buffN[0], nx, MPI_DOUBLE, domain.neighbour_north, domain.rank,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_south>=0) {
        // set tag to be the sender's rank
        // post receive
        MPI_Irecv(&bndS[0], nx, MPI_DOUBLE, domain.neighbour_south, domain.neighbour_south,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;

        // pack south buffer
        for(int i=0; i<nx; i++)
            buffS[i] = U(i,0);

        // post send
        MPI_Isend(&buffS[0], nx, MPI_DOUBLE, domain.neighbour_south, domain.rank,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_east>=0) {
        // set tag to be the sender's rank
        // post receive
        MPI_Irecv(&bndE[0], ny, MPI_DOUBLE, domain.neighbour_east, domain.neighbour_east,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;

        // pack north buffer
        for(int j=0; j<ny; j++)
            buffE[j] = U(nx-1,j);

        // post send
        MPI_Isend(&buffE[0], ny, MPI_DOUBLE, domain.neighbour_east, domain.rank,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;
    }
    if(domain.neighbour_west>=0) {
        // set tag to be the sender's rank
        // post receive
        MPI_Irecv(&bndW[0], ny, MPI_DOUBLE, domain.neighbour_west, domain.neighbour_west,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;

        // pack north buffer
        for(int j=0; j<ny; j++)
            buffW[j] = U(0,j);

        // post send
        MPI_Isend(&buffW[0], ny, MPI_DOUBLE, domain.neighbour_west, domain.rank,
            MPI_COMM_WORLD, requests+num_requests);
        num_requests++;
    }

    // wait on the receives
    MPI_Waitall(num_requests, requests, statuses);


	diffusion(U.ptr_, S.ptr_);


    
    // Accumulate the flop counts
    // 8 ops total per point
    stats::flops_diff +=
        + 12 * (nx - 2) * (ny - 2) // interior points
        + 11 * (nx - 2  +  ny - 2) // NESW boundary points
        + 11 * 4;                                  // corner points
}

} // namespace operators
