//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

__global__ void diffusion(const double* up, double* sp);

#define cudaMallocDevice(dst, size) {                                                                          \
	double* memPtr = NULL;                                                                                     \
	CUDA_ERR_CHECK(cudaMalloc(&memPtr, size));                                                                 \
	double* ptrPtr;                                                                                            \
	CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptrPtr, gpu::dst));                                           \
	CUDA_ERR_CHECK(cudaMemcpy(ptrPtr, &memPtr, sizeof(double*), cudaMemcpyHostToDevice));                      \
	CUDA_ERR_CHECK(cudaMemcpy(memPtr, cpu::dst, size, cudaMemcpyHostToDevice));                                \
}

template<typename T>
inline T get_device_value(T& var)
{
	T* ptr;
	CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptr, var));
	T value;
	CUDA_ERR_CHECK(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost));
	return value;
}

#endif // OPERATORS_H

