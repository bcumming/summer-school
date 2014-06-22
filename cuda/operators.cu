//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "check.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

#include <thrust/extrema.h>

using namespace thrust::system::cuda::detail;

#define U(j,i)    up[(i) + (j)*nx]
#define S(j,i)    sp[(i) + (j)*nx]
#define X(j,i) x_old[(i) + (j)*nx]

namespace gpu
{
    __global__ void diffusion_interior_grid_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;
        int    iend  = options.nx - 1;
        int    jend  = options.ny - 1;

        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        
        if ((i >= iend) || (j >= jend)) return;

        int nx = options.nx;

        S(j, i) = -(4. + alpha) * U(j,i)              // central point
                                + U(j,i-1) + U(j,i+1) // east and west
                                + U(j-1,i) + U(j+1,i) // north and south

                                + alpha * X(j,i)
                                + dxs * U(j,i) * (1.0 - U(j,i));
    }
    
    __global__ void diffusion_east_boundary_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;

        int i = options.nx - 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

        int nx = options.nx;

        S(j, i) = -(4. + alpha) * U(j,i)
                                + U(j, i - 1) + U(j - 1, i) + U(j + 1, i)

                                + alpha * X(j, i) + bndE[j]
                                + dxs * U(j, i) * (1.0 - U(j, i));
    }

    __global__ void diffusion_west_boundary_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;

        int i = 0;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

        int nx = options.nx;

        S(j, i) = -(4. + alpha) * U(j, i)
                                + U(j, i + 1) + U(j - 1, i) + U(j + 1, i)

                                + alpha * X(j, i) + bndW[j]
                                + dxs * U(j, i) * (1.0 - U(j, i));
    }

    __global__ void diffusion_north_boundary_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;

        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = options.ny - 1;

        int nx = options.nx;

        S(j, i) = -(4. + alpha) * U(j, i)
                                + U(j, i - 1) + U(j, i + 1) + U(j - 1, i)

                                + alpha * X(j, i) + bndN[i]
                                + dxs * U(j, i) * (1.0 - U(j, i));
    }

    __global__ void diffusion_south_boundary_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;

        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = 0;

        int nx = options.nx;

        S(j, i) = -(4. + alpha) * U(j, i)
                                + U(j, i - 1) + U(j, i + 1) + U(j + 1, i)

                                + alpha * X(j, i) + bndS[i]
                                + dxs * U(j, i) * (1.0 - U(j, i));
    }

    __global__ void diffusion_corner_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000. * options.dx * options.dx;
        double alpha = options.alpha;

        int nx = options.nx;

        {
            int j = options.ny - 1;
            {
                int i = 0; // NW corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i + 1) + U(j - 1, i)

                            + alpha * X(j, i) + bndW[j] + bndN[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
            {
                int i = options.nx - 1; // NE corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j - 1, i)
                            + alpha * X(j, i) + bndE[j] + bndN[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
        }
        {
            int j = 0;
            {
                int i = 0; // SW corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i + 1) + U(j + 1, i)
                            + alpha * X(j, i) + bndW[j] + bndS[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
            {
                int i = options.nx - 1; // SE corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j + 1, i)
                            + alpha * X(j, i) + bndE[j] + bndS[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
        }
    }
}

static inline void grid_block(
    size_t szblock, int nx, int ny, dim3& grid, dim3& block)
{
    grid = dim3(1, 1, 1);
    block = dim3(1, 1, 1);

    if (szblock > nx)
    {
        block.x = nx;
        block.y = min(ny, (int)szblock / block.x);
        grid.y = ny / block.y;
        if (ny % block.y) grid.y++;
    }
    else
    {
        block.x = szblock;
        grid.x = nx / block.x;
        if (nx % block.x) grid.x++;
        grid.y = ny;
    }
}

void diffusion(const double* up, double* sp)
{
    using namespace cpu;

    int nx = options.nx;
    int ny = options.ny;

    // Allocate temporary GPU copies for "up" and "sp".
    double *up_gpu = NULL, *sp_gpu = NULL;
    CUDA_ERR_CHECK(cudaMalloc(&up_gpu, sizeof(double) * nx * ny));
    CUDA_ERR_CHECK(cudaMalloc(&sp_gpu, sizeof(double) * nx * ny));

    // Copy actual "up" data to GPU.
    CUDA_ERR_CHECK(cudaMemcpy(up_gpu, up, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    // Copy actual x_old data to GPU.
    double** gpu_x_old;
    CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&gpu_x_old, gpu::x_old));
    double* gpu_x_old_value;
    CUDA_ERR_CHECK(cudaMemcpy(&gpu_x_old_value, gpu_x_old, sizeof(double*), cudaMemcpyDeviceToHost));
    CUDA_ERR_CHECK(cudaMemcpy(gpu_x_old_value, cpu::x_old, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
    
    // Use Thrust occupancy calculator to determine the
    // best size of block.
    struct function_attributes_t attrs;
    {
        cudaFuncAttributes funcAttrs;
        CUDA_ERR_CHECK(cudaFuncGetAttributes(&funcAttrs, gpu::diffusion_interior_grid_points));
        attrs.constSizeBytes = funcAttrs.constSizeBytes;
        attrs.localSizeBytes = funcAttrs.localSizeBytes;
        attrs.maxThreadsPerBlock = funcAttrs.maxThreadsPerBlock;
        attrs.numRegs = funcAttrs.numRegs;
        attrs.sharedSizeBytes = funcAttrs.sharedSizeBytes;
    }
    struct device_properties_t props;
    {
        cudaDeviceProp gpuProps;
        CUDA_ERR_CHECK(cudaGetDeviceProperties(&gpuProps, 0));
        props.major = gpuProps.major;
        memcpy(&props.maxGridSize, &gpuProps.maxGridSize, sizeof(int) * 3);
        props.maxThreadsPerBlock = gpuProps.maxThreadsPerBlock;
        props.maxThreadsPerMultiProcessor = gpuProps.maxThreadsPerMultiProcessor;
        props.minor = gpuProps.minor;
        props.multiProcessorCount = gpuProps.multiProcessorCount;
        props.regsPerBlock = gpuProps.regsPerBlock;
        props.sharedMemPerBlock = gpuProps.sharedMemPerBlock;
        props.warpSize = gpuProps.warpSize;
    }
    size_t szblock = block_size_with_maximum_potential_occupancy(attrs, props);

    // Launch kernel for parallel processing of interior points.
    dim3 grid, block;
    grid_block(szblock, nx - 2, ny - 2, grid, block);
    gpu::diffusion_interior_grid_points<<<grid, block>>>(up_gpu, sp_gpu);
    
    // Launch kernels for parallel processing of boundary points.
    grid_block(szblock, 1, ny - 2, grid, block);
    gpu::diffusion_east_boundary_points<<<grid, block>>>(up_gpu, sp_gpu);
    grid_block(szblock, 1, ny - 2, grid, block);
    gpu::diffusion_west_boundary_points<<<grid, block>>>(up_gpu, sp_gpu);
    grid_block(szblock, nx - 2, 1, grid, block);
    gpu::diffusion_north_boundary_points<<<grid, block>>>(up_gpu, sp_gpu);
    grid_block(szblock, nx - 2, 1, grid, block);
    gpu::diffusion_south_boundary_points<<<grid, block>>>(up_gpu, sp_gpu);
    
    // Launch kernel for single-threaded processing of corner points.
    gpu::diffusion_corner_points<<<1, 1>>>(up_gpu, sp_gpu);
    
    // Copy back the resulting "sp" data.
    CUDA_ERR_CHECK(cudaMemcpy(sp, sp_gpu, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));
    
    // Free temporary GPU arrays.
    CUDA_ERR_CHECK(cudaFree(up_gpu));
    CUDA_ERR_CHECK(cudaFree(sp_gpu));

    // Accumulate the flop counts
    // 8 ops total per point
    flops_diff +=
        + 12 * (options.nx - 2) * (options.ny - 2) // interior points
        + 11 * (options.nx - 2  +  options.ny - 2) // NESW boundary points
        + 11 * 4;                                  // corner points}
}

