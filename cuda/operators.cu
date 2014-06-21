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
    // the interior grid points (nx - 2) * (ny - 2)
    __global__ void diffusion_interior_grid_points(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000.*options.dx*options.dx;
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
    
    __global__ void diffusion(const double* up, double* sp)
    {
        using namespace gpu;

        double dxs   = 1000.*options.dx*options.dx;
        double alpha = options.alpha;
        int    iend  = options.nx - 1;
        int    jend  = options.ny - 1;

        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

        int nx = options.nx;

        // the east boundary
        {
            i = options.nx - 1;
            for (j = 1; j < jend; j++)
            {
                S(j, i) = -(4. + alpha) * U(j,i)
                            + U(j, i - 1) + U(j - 1, i) + U(j + 1, i)
                            + alpha*X(j, i) + bndE[j]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
        }
        // the west boundary
        {
            i = 0;
            for (j = 1; j < jend; j++)
            {
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i + 1) + U(j - 1, i) + U(j + 1, i)

                            + alpha*X(j, i) + bndW[j]
                            + dxs*U(j, i) * (1.0 - U(j, i));
            }
        }
        // the north boundary (plus NE and NW corners)
        {
            j = options.ny - 1;

            {
                i = 0; // NW corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i + 1) + U(j - 1, i)

                            + alpha * X(j, i) + bndW[j] + bndN[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }

            // north boundary
            for (i = 1; i < iend; i++)
            {
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j, i + 1) + U(j - 1, i)
                            + alpha*X(j, i) + bndN[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }

            {
                i = options.nx; // NE corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j - 1, i)
                            + alpha * X(j, i) + bndE[j] + bndN[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
        }
        // the south boundary
        {
            j = 0;
            {
                i = 0; // SW corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i + 1) + U(j + 1, i)
                            + alpha * X(j, i) + bndW[j] + bndS[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
            // south boundary
            for (i = 1; i < iend; i++)
            {
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j, i + 1) + U(j + 1, i)
                            + alpha * X(j, i) + bndS[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
            //
            {
                i = options.nx - 1; // SE corner
                S(j, i) = -(4. + alpha) * U(j, i)
                            + U(j, i - 1) + U(j + 1, i)
                            + alpha * X(j, i) + bndE[j] + bndS[i]
                            + dxs * U(j, i) * (1.0 - U(j, i));
            }
        }
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
    dim3 grid(1, 1, 1), block(1, 1, 1);
    if (szblock > nx - 2)
    {
    	block.x = nx - 2;
    	block.y = szblock / block.x;
    	grid.y = (ny - 2) / block.y;
    	if ((ny - 2) % block.y) grid.y++;
    }
    else
    {
    	block.x = szblock;
    	grid.x = (nx - 2) / block.x;
    	if ((nx - 2) % block.x) grid.x++;
    	grid.y = ny - 2;
    }
    
    // Launch kernel for parallel processing of interior points.
    gpu::diffusion_interior_grid_points<<<grid, block>>>(up_gpu, sp_gpu);
    
    // Launch kernel for single-threaded processing of boundary points.
    gpu::diffusion<<<1, 1>>>(up_gpu, sp_gpu);
    
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

