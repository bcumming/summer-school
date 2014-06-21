#include "stats.h"

unsigned long long cpu::flops_diff, cpu::flops_bc, cpu::flops_blas1;
unsigned int cpu::iters_cg, cpu::iters_newton;
int cpu::verbose_output;

unsigned long long gpu::flops_diff, gpu::flops_bc, gpu::flops_blas1;
unsigned int gpu::iters_cg, gpu::iters_newton;
int gpu::verbose_output;
