#include "stats.h"

unsigned long long gpu::flops_diff, gpu::flops_blas1;
unsigned int gpu::iters_cg, gpu::iters_newton;
int gpu::verbose_output = 0;

unsigned long long gpu::flops_diff_d = 0, gpu::flops_blas1_d = 0;
unsigned int gpu::iters_cg_d = 0, gpu::iters_newton_d = 0;
