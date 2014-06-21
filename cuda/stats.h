#ifndef STATS_H
#define STATS_H

namespace cpu
{
	extern unsigned long long flops_diff, flops_bc, flops_blas1;
	extern unsigned int iters_cg, iters_newton;
	extern int verbose_output;
}

namespace gpu
{
	extern __device__ unsigned long long flops_diff, flops_bc, flops_blas1;
	extern __device__ unsigned int iters_cg, iters_newton;
	extern __constant__ int verbose_output;
}

#endif // STATS_H

