// *****************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

#include <thrust/extrema.h>

#define cudaMallocDevice(dst, size) {                                                                          \
	double* memPtr = NULL;                                                                                     \
	CUDA_ERR_CHECK(cudaMalloc(&memPtr, size));                                                                 \
	double* ptrPtr;                                                                                            \
	CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptrPtr, gpu::dst));                                           \
	CUDA_ERR_CHECK(cudaMemcpy(ptrPtr, &memPtr, sizeof(double*), cudaMemcpyHostToDevice));                      \
	CUDA_ERR_CHECK(cudaMemcpy(memPtr, cpu::dst, size, cudaMemcpyHostToDevice));                                \
}

namespace gpu
{
	__device__ void diffusion(const double* up, double* sp);

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
	__device__ inline size_t get_optimal_szblock(T kernel)
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
		return block_size_with_maximum_potential_occupancy(attrs, props);
	}

	template<typename T>
	__device__ inline void get_optimal_grid_block_config(T kernel,
		int nx, int ny, dim3& grid, dim3& blocks)
	{
		size_t szblock = get_optimal_szblock(kernel);

		grid.x = 1; grid.y = 1; grid.z = 1;
		blocks.x = 1; blocks.y = 1; blocks.z = 1;

		if (szblock > nx)
		{
			blocks.x = nx;
			blocks.y = min(ny, (int)szblock / blocks.x);
			grid.y = ny / blocks.y;
			if (ny % blocks.y) grid.y++;
		}
		else
		{
			blocks.x = szblock;
			grid.x = nx / blocks.x;
			if (nx % blocks.x) grid.x++;
			grid.y = ny;
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
}

#endif // OPERATORS_H

