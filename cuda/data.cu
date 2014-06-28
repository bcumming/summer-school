#include "data.h"

#include <stdio.h>

struct discretization_t cpu::options;

cudaDeviceProp cpu::props;

// fields that hold the solution
__device__ double * __restrict__ gpu::x_old = NULL; // 2d
__device__ double * __restrict__ gpu::bndN = NULL, * __restrict__ gpu::bndE = NULL;
__device__ double * __restrict__ gpu::bndS = NULL, * __restrict__ gpu::bndW = NULL;

__constant__ struct discretization_t gpu::options;

