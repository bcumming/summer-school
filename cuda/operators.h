// *****************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

namespace gpu
{
	namespace diffusion_interior_grid_points_kernel
	{
		__global__ void kernel(const double* up, double* sp);

		extern __constant__ config_t config;
	}

	namespace diffusion_east_west_boundary_points_kernel
	{
		__global__ void kernel(const double* up, double* sp);

		extern __constant__ config_t config;
	}

	namespace diffusion_north_south_boundary_points_kernel
	{
		__global__ void kernel(const double* up, double* sp);

		extern __constant__ config_t config;
	}

	__device__ void diffusion(const double* up, double* sp);
}

#endif // OPERATORS_H

