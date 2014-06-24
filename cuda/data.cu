#include "data.h"

#include <stdio.h>

// fields that hold the solution
double *cpu::x_old = NULL; // 2d
double *cpu::bndN = NULL, *cpu::bndE = NULL, *cpu::bndS = NULL, *cpu::bndW = NULL;
double *cpu::b = NULL, *cpu::deltax = NULL;

struct discretization_t cpu::options;

// fields that hold the solution
__device__ double *gpu::x_old = NULL; // 2d
__device__ double *gpu::bndN = NULL, *gpu::bndE = NULL, *gpu::bndS = NULL, *gpu::bndW = NULL;
__device__ double *gpu::b = NULL, *gpu::deltax = NULL;

__constant__ struct discretization_t gpu::options;

__constant__ cudaDeviceProp gpu::props;
