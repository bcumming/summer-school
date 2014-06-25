#include "stats.h"

unsigned long long cpu::flops_diff = 0, cpu::flops_bc = 0, cpu::flops_blas1 = 0;
unsigned int cpu::iters_cg = 0, cpu::iters_newton = 0;
int cpu::verbose_output = 0;

unsigned long long gpu::flops_diff = 0, gpu::flops_bc = 0, gpu::flops_blas1 = 0;
unsigned int gpu::iters_cg = 0, gpu::iters_newton = 0;
int gpu::verbose_output = 0;
