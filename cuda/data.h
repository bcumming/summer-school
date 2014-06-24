#ifndef DATA_H
#define DATA_H

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct discretization_t
{
	int nx;       // x dimension
	int ny;       // y dimension
	int nt;       // number of time steps
	int N;        // total number of grid points
	double dt;    // time step size
	double dx;    // distance between grid points
	double alpha; // dx^2/(D*dt)
};

namespace cpu
{
	extern struct discretization_t options;

	extern cudaDeviceProp props;
}

namespace gpu
{
	// fields that hold the solution
	extern __device__ double *x_old; // 2d
	extern __device__ double *bndN, *bndE, *bndS, *bndW; // 1d

	extern __constant__ struct discretization_t options;
}

#include <thrust/extrema.h>

namespace gpu
{
	// We redefine dim3 under namespace, because the default one has
	// constructors, which is not allowed for types device variables
	// (dim3 is used as device vars type below to keep kernel compute
	// grid configuration).
	struct dim3
	{
		unsigned int x, y, z;
		
		__device__ operator ::dim3()
		{
			return ::dim3(x, y, z);
		}
	};
	
	// Use Thrust occupancy calculator to determine the best size of block.
	template<typename T>
	inline size_t get_optimal_szblock(T kernel)
	{
		using namespace gpu;
		using namespace thrust::system::cuda::detail;

		struct function_attributes_t attrs;
		{
			cudaFuncAttributes funcAttrs;
			CUDA_ERR_CHECK(cudaFuncGetAttributes(&funcAttrs, kernel));
			attrs.constSizeBytes = funcAttrs.constSizeBytes;
			attrs.localSizeBytes = funcAttrs.localSizeBytes;
			attrs.maxThreadsPerBlock = funcAttrs.maxThreadsPerBlock;
			attrs.numRegs = funcAttrs.numRegs;
			attrs.sharedSizeBytes = funcAttrs.sharedSizeBytes;
		}
		struct device_properties_t props;
		{
			props.major = cpu::props.major;
			memcpy(&props.maxGridSize, &cpu::props.maxGridSize, sizeof(int) * 3);
			props.maxThreadsPerBlock = cpu::props.maxThreadsPerBlock;
			props.maxThreadsPerMultiProcessor = cpu::props.maxThreadsPerMultiProcessor;
			props.minor = cpu::props.minor;
			props.multiProcessorCount = cpu::props.multiProcessorCount;
			props.regsPerBlock = cpu::props.regsPerBlock;
			props.sharedMemPerBlock = cpu::props.sharedMemPerBlock;
			props.warpSize = cpu::props.warpSize;
		}
		return block_size_with_maximum_potential_occupancy(attrs, props);
	}

	template<typename T>
	inline void get_optimal_grid_block_config(T kernel,
		int nx, int ny, dim3* grid, dim3* blocks)
	{
		size_t szblock = get_optimal_szblock(kernel);

		grid->x = 1; grid->y = 1; grid->z = 1;
		blocks->x = 1; blocks->y = 1; blocks->z = 1;

		if (szblock > nx)
		{
			blocks->x = nx;
			blocks->y = min(ny, (int)szblock / blocks->x);
			grid->y = ny / blocks->y;
			if (ny % blocks->y) grid->y++;
		}
		else
		{
			blocks->x = szblock;
			grid->x = nx / blocks->x;
			if (nx % blocks->x) grid->x++;
			grid->y = ny;
		}
	}

	template<typename T>
	inline T get_value(T& var)
	{
		T* ptr;
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptr, var));
		T value;
		CUDA_ERR_CHECK(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost));
		return value;
	}

	typedef struct __attribute__((packed)) { dim3 grid, block; } config_t;
}

#endif // DATA_H

