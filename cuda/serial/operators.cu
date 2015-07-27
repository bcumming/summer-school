//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "cuda_helpers.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

// POD type holding information for device
struct DiffusionParams {
    int nx;
    int ny;
    double alpha;
    double dxs;
    double *x_old;
    double *bndN;
    double *bndE;
    double *bndS;
    double *bndW;
};

// copy of global parameters for kernels to use directly
__device__
DiffusionParams params;

// copies the global parameters to the device
// must be called once at setup, before any of the stencil kernels are executed
void setup_params_on_device(
        int nx, int ny,
        double alpha, double dxs)
{
    auto p = DiffusionParams {
        nx,
        ny,
        alpha,
        dxs,
        data::x_old.device_data(),
        data::bndN.device_data(),
        data::bndE.device_data(),
        data::bndS.device_data(),
        data::bndW.device_data()
    };

    cuda_check_status(
        cudaMemcpyToSymbol(params, &p, sizeof(DiffusionParams))
    );
}

namespace kernels {
    __global__
    void stencil_interior(double* S, const double *U) {
        auto nx = params.nx;
        auto ny = params.ny;

        auto i = threadIdx.x + blockDim.x*blockIdx.x;
        auto j = threadIdx.y + blockDim.y*blockIdx.y;
        auto pos = i + j * nx;

        // stencil is applied to interior grid pints, i.e. (i,j) such that
        //      i \in [1, nx-1)
        //      j \in [1, ny-1)
        auto is_interior =
            i<(nx-1) && j<(ny-1) && (i>1 && j>1);
        if(is_interior) {
            S[pos] = -(4. + params.alpha) * U[pos]          // central point
                                   + U[pos-1] + U[pos+1]    // east and west
                                   + U[pos-nx] + U[pos+nx]  // north and south
                                   + params.alpha * params.x_old[pos]
                                   + params.dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_east_west(double* S, const double *U) {
        auto j = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        if(j>0 && j<ny-1) {
            // EAST : i = nx-1
            auto pos = find_pos(nx-1, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos-nx] + U[pos+nx]
                        + alpha*params.x_old[pos] + params.bndE[j]
                        + dxs * U[pos] * (1.0 - U[pos]);

            // WEST : i = 0
            pos = find_pos(0, j);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos+1] + U[pos-ny] + U[pos+nx]
                        + alpha * params.x_old[pos] + params.bndW[j]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_north_south(double* S, const double *U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        if(i>0 && i<nx-1) {
            // NORTH : j = ny -1
            auto pos = i + nx*(ny-1);
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos-nx]
                        + alpha*params.x_old[pos] + params.bndN[i]
                        + dxs * U[pos] * (1.0 - U[pos]);
            // SOUTH : j = 0
            pos = i;
            S[pos] = -(4. + alpha) * U[pos]
                        + U[pos-1] + U[pos+1] + U[pos+nx]
                        + alpha * params.x_old[pos] + params.bndS[i]
                        + dxs * U[pos] * (1.0 - U[pos]);
        }
    }

    __global__
    void stencil_corners(double* S, const double* U) {
        auto i = threadIdx.x + blockDim.x*blockIdx.x;

        auto nx = params.nx;
        auto ny = params.ny;
        auto alpha = params.alpha;
        auto dxs = params.dxs;

        auto find_pos = [&nx] (size_t i, size_t j) {
            return i + j * nx;
        };

        // only 1 thread executes this kernel
        if(i==0) {
            // NORTH-EAST
            auto pos = find_pos(nx-1, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[ny-1] // east and west
                                   + U[pos-nx] + params.bndN[nx-1] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-EAST
            pos = find_pos(nx-1, 0);
            S[pos] = -(4. + alpha) * U[pos]                     // central point
                                   + U[pos-1]    + params.bndE[0]      // east and west
                                   + params.bndS[nx-1]+ U[pos+nx]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // SOUTH-WEST
            pos = find_pos(0, 0);
            S[pos] = -(4. + alpha) * U[pos]                // central point
                                   + params.bndW[0] + U[pos+1]    // east and west
                                   + params.bndS[0] + U[pos+nx] // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);

            // NORTH-WEST
            pos = find_pos(0, ny-1);
            S[pos] = -(4. + alpha) * U[pos]                 // central point
                                   + params.bndW[nx-1]+ U[pos+1] // east and west
                                   + U[pos-nx] + params.bndN[0]  // north and south
                                   + alpha * params.x_old[pos]
                                   + dxs * U[pos] * (1.0 - U[pos]);
        }
    }
}

enum class Boundary {north, east, south, west};

void diffusion(data::Field const& U, data::Field &S)
{
    using data::options;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;


    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;

    // calculates the linear index into an array of width nx
    // from an (i,j) coordinate pair
    auto idx = [&nx] (size_t i, size_t j) {
        return i + j * nx;
    };

    static bool is_initialized = false;
    if(!is_initialized) {
        setup_params_on_device(nx, ny, alpha, dxs);
        is_initialized = true;
    }

    // apply stencil to the interior grid points
    auto calculate_grid_dim = [] (size_t n, size_t block_dim) {
        return n/block_dim + (n%block_dim ? 1 : 0);
    };
    dim3 block_dim(16, 16); // use 8x8 thread block dimensions
    dim3 grid_dim(
        calculate_grid_dim(nx, block_dim.x),
        calculate_grid_dim(ny, block_dim.y)
    );

    kernels::stencil_interior<<<grid_dim, block_dim>>>(S.device_data(), U.device_data());
    cuda_check_last_kernel("interior point stencil kernel launch");

    // apply stencil at boundaries
    auto bnd_grid_dim_y = calculate_grid_dim(ny, 64);
    kernels::stencil_east_west<<<bnd_grid_dim_y, 64>>>(S.device_data(), U.device_data());
    cuda_check_last_kernel("east-west stencil kernel");

    auto bnd_grid_dim_x = calculate_grid_dim(nx, 64);
    //std::cout << "[[ " << 64 << ", " << bnd_grid_dim_x << "]]" << std::endl;
    kernels::stencil_north_south<<<bnd_grid_dim_x, 64>>>(S.device_data(), U.device_data());
    cuda_check_last_kernel("north-south stencil kernel");

    kernels::stencil_corners<<<1, 1>>>(S.device_data(), U.device_data());
    cuda_check_last_kernel("corner stencil kernel");

    stats::flops_diff +=
        + 12 * (nx - 2) * (ny - 2) // interior points
        + 11 * (nx - 2  +  ny - 2) // NESW boundary points
        + 11 * 4;                  // corner points
}

} // namespace operators
