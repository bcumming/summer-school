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

#define diffusion_load(up, sp)                                                                                 \
{                                                                                                              \
	int nx = cpu::options.nx;                                                                                  \
	int ny = cpu::options.ny;                                                                                  \
		                                                                                                       \
    /* Copy actual "up" data to GPU. */                                                                        \
    double* cpu_up = NULL;                                                                                     \
   	cpu_up = cpu::up;                                                                                          \
  	double** gpu_up;                                                                                           \
    CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&gpu_up, gpu::up));                                            \
    double* gpu_up_value;                                                                                      \
    CUDA_ERR_CHECK(cudaMemcpy(&gpu_up_value, gpu_up, sizeof(double*), cudaMemcpyDeviceToHost));                \
    CUDA_ERR_CHECK(cudaMemcpy(gpu_up_value, cpu_up, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));        \
                                                                                                               \
    /* Copy actual x_old data to GPU. */                                                                       \
    double** gpu_x_old;                                                                                        \
    CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&gpu_x_old, gpu::x_old));                                      \
    double* gpu_x_old_value;                                                                                   \
    CUDA_ERR_CHECK(cudaMemcpy(&gpu_x_old_value, gpu_x_old, sizeof(double*), cudaMemcpyDeviceToHost));          \
    CUDA_ERR_CHECK(cudaMemcpy(gpu_x_old_value, cpu::x_old, sizeof(double) * nx * ny, cudaMemcpyHostToDevice)); \
}

#define diffusion_unload(up, sp)                                                                               \
{                                                                                                              \
	int nx = cpu::options.nx;                                                                                  \
	int ny = cpu::options.ny;                                                                                  \
                                                                                                               \
    /* Copy back the resulting "sp" data. */                                                                   \
    double* cpu_sp = NULL;                                                                                     \
   	cpu_sp = cpu::sp;                                                                                          \
    double** gpu_sp;                                                                                           \
    CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&gpu_sp, gpu::sp));                                            \
    double* gpu_sp_value;                                                                                      \
    CUDA_ERR_CHECK(cudaMemcpy(&gpu_sp_value, gpu_sp, sizeof(double*), cudaMemcpyDeviceToHost));                \
    CUDA_ERR_CHECK(cudaMemcpy(cpu_sp, gpu_sp_value, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));        \
}

inline double* get_device_value(double*& var)
{
	double** ptr;
	CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptr, var));
	double* value;
	CUDA_ERR_CHECK(cudaMemcpy(&value, ptr, sizeof(double*), cudaMemcpyDeviceToHost));
	return value;
}

#endif // OPERATORS_H

