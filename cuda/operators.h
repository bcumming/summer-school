// *****************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

#include "check.h"
#include "data.h"
#include "stats.h"

#define U(j,i)    up[(i) + (j)*nx]
#define S(j,i)    sp[(i) + (j)*nx]
#define X(j,i) x_old[(i) + (j)*nx]

namespace gpu
{
	namespace diffusion_interior_grid_points_kernel
	{
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
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

		__constant__ config_t config_c;
		__shared__ config_t config;
	}

	namespace diffusion_east_west_boundary_points_kernel
	{
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
			using namespace gpu;

			double dxs   = 1000. * options.dx * options.dx;
			double alpha = options.alpha;
			int    jend  = options.ny - 1;

			int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

			if (j >= jend) return;

			int nx = options.nx;

			int i = options.nx - 1;

			S(j, i) = -(4. + alpha) * U(j,i)
					                + U(j, i - 1) + U(j - 1, i) + U(j + 1, i)

					                + alpha * X(j, i) + bndE[j]
					                + dxs * U(j, i) * (1.0 - U(j, i));

			i = 0;

			S(j, i) = -(4. + alpha) * U(j, i)
					                + U(j, i + 1) + U(j - 1, i) + U(j + 1, i)

					                + alpha * X(j, i) + bndW[j]
					                + dxs * U(j, i) * (1.0 - U(j, i));
		}

		__constant__ config_t config_c;
		__shared__ config_t config;
	}

	namespace diffusion_north_south_boundary_points_kernel
	{
		__global__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
		{
			using namespace gpu;

			double dxs   = 1000. * options.dx * options.dx;
			double alpha = options.alpha;
			int    iend  = options.nx - 1;

			int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

			if (i >= iend) return;

			int nx = options.nx;

			int j = options.ny - 1;

			S(j, i) = -(4. + alpha) * U(j, i)
					                + U(j, i - 1) + U(j, i + 1) + U(j - 1, i)

					                + alpha * X(j, i) + bndN[i]
					                + dxs * U(j, i) * (1.0 - U(j, i));

			j = 0;

			S(j, i) = -(4. + alpha) * U(j, i)
					                + U(j, i - 1) + U(j, i + 1) + U(j + 1, i)

					                + alpha * X(j, i) + bndS[i]
					                + dxs * U(j, i) * (1.0 - U(j, i));
		}

		__constant__ config_t config_c;
		__shared__ config_t config;
	}

	namespace diffusion_corner_points_kernel
	{
		inline __device__ void kernel(const double* const __restrict__ up, double* __restrict__ sp)
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

	inline __device__ void diffusion(const double* const __restrict__ up, double* __restrict__ sp)
	{
		using namespace gpu;

		// Launch kernel for parallel processing of interior points.
		{
			using namespace diffusion_interior_grid_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<<<config.grid, config.block>>>(up, sp));
		}
	
		// Launch kernels for parallel processing of boundary points.
		{
			using namespace diffusion_east_west_boundary_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<<<config.grid, config.block>>>(up, sp));
		}
		{
			using namespace diffusion_north_south_boundary_points_kernel;
			CUDA_LAUNCH_ERR_CHECK(kernel<<<config.grid, config.block>>>(up, sp));
		}
	
		// Finally, single-threaded processing of corner points.
		diffusion_corner_points_kernel::kernel(up, sp);
	
		// Accumulate the flop counts
		// 8 ops total per point
		flops_diff +=
			+ 12 * (options.nx - 2) * (options.ny - 2) // interior points
			+ 11 * (options.nx - 2  +  options.ny - 2) // NESW boundary points
			+ 11 * 4;                                  // corner points}
	}
}

#endif // OPERATORS_H

