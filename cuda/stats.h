#ifndef STATS_H
#define STATS_H

namespace gpu
{
	extern __shared__ unsigned long long flops_diff, flops_blas1;
	extern __shared__ unsigned int iters_cg, iters_newton;
	extern __constant__ int verbose_output;

	extern __device__ unsigned long long flops_diff_d, flops_blas1_d;
	extern __device__ unsigned int iters_cg_d, iters_newton_d;
}

#endif // STATS_H

