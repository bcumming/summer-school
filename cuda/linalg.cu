// linear algebra subroutines
// Ben Cumming @ CSCS

#include "check.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace
{
	__device__ int cg_initialized = 0;
	__device__ double *r = NULL, *Ap = NULL, *p = NULL;
	__device__ double *Fx = NULL, *Fxold = NULL, *v = NULL, *xold = NULL; // 1d

	// initialize temporary storage fields used by the cg solver
	// I do this here so that the fields are persistent between calls
	// to the CG solver. This is useful if we want to avoid malloc/free calls
	// on the device for the OpenACC implementation (feel free to suggest a better
	// method for doing this)
	__device__ void cg_init(const int N)
	{
		Ap    = (double*)malloc(sizeof(double) * N);
		r     = (double*)malloc(sizeof(double) * N); 
		p     = (double*)malloc(sizeof(double) * N);
		Fx    = (double*)malloc(sizeof(double) * N);
		Fxold = (double*)malloc(sizeof(double) * N);
		v     = (double*)malloc(sizeof(double) * N);
		xold  = (double*)malloc(sizeof(double) * N);

		cg_initialized = 1;
	}
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

namespace gpu
{
	namespace ss_dot_kernel
	{
		// computes the inner product of x and y
		// x and y are vectors of length N
		__global__ void kernel(const double* x, const double* y, double* result)
		{
			extern __shared__ double shared[];

			// Each block hanldes (2 * blockDim.x) elements of reduction.
			int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

			// Load product of first 2 pairs into shared memory:
			// idx-th and (idx + blockDim.x)-th.
			shared[threadIdx.x] = x[i] * y[i] + x[i + blockDim.x] * y[i + blockDim.x];
	
			__syncthreads();

			// Reduce pairs in shared memory.
			for (int s = blockDim.x / 2; s > 32; s >>= 1)
			{
				if (threadIdx.x < s)
					shared[threadIdx.x] += shared[threadIdx.x + s];
		
				__syncthreads();
			}
	
			// Unroll last 32 iterations of loop.
			// There is no need for synchronizations, since all accesses
			// are within single warp.
			if (threadIdx.x < 32)
			{
				volatile double* vshared = shared;
				vshared[threadIdx.x] += vshared[threadIdx.x + 32];
				vshared[threadIdx.x] += vshared[threadIdx.x + 16];
				vshared[threadIdx.x] += vshared[threadIdx.x +  8];
				vshared[threadIdx.x] += vshared[threadIdx.x +  4];
				vshared[threadIdx.x] += vshared[threadIdx.x +  2];
				vshared[threadIdx.x] += vshared[threadIdx.x +  1];
			}

			// The first thread writes the result.
			if (threadIdx.x == 0)
				result[blockIdx.x] = shared[0];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
		__device__ double* buffer = NULL;
	}
}

__device__ double ss_dot(const double* x, const double* y, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_dot_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N / 2, 1, grid, block);
		grid_block_init = true;
	}

	if (!buffer)
		CUDA_ERR_CHECK(cudaMalloc(&buffer, sizeof(double) * grid.x));

	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block, block.x * sizeof(double)>>>(x, y, buffer));
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	double result = buffer[0];
	for (int i = 1; i < grid.x; i++)
		result += buffer[i];
	
    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

namespace gpu
{
	namespace ss_sum_kernel
	{
		// computes the sum of x elements
		// x is a vector of length N
		__global__ void kernel(const double* x, double* result)
		{
			extern __shared__ double shared[];

			// Each block hanldes (2 * blockDim.x) elements of reduction.
			int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

			// Load product of first 2 pairs into shared memory:
			// idx-th and (idx + blockDim.x)-th.
			shared[threadIdx.x] = x[i] + x[i + blockDim.x];
	
			__syncthreads();

			// Reduce pairs in shared memory.
			for (int s = blockDim.x / 2; s > 32; s >>= 1)
			{
				if (threadIdx.x < s)
					shared[threadIdx.x] += shared[threadIdx.x + s];
		
				__syncthreads();
			}
	
			// Unroll last 32 iterations of loop.
			// There is no need for synchronizations, since all accesses
			// are within single warp.
			if (threadIdx.x < 32)
			{
				volatile double* vshared = shared;
				vshared[threadIdx.x] += vshared[threadIdx.x + 32];
				vshared[threadIdx.x] += vshared[threadIdx.x + 16];
				vshared[threadIdx.x] += vshared[threadIdx.x +  8];
				vshared[threadIdx.x] += vshared[threadIdx.x +  4];
				vshared[threadIdx.x] += vshared[threadIdx.x +  2];
				vshared[threadIdx.x] += vshared[threadIdx.x +  1];
			}

			// The first thread writes the result.
			if (threadIdx.x == 0)
				result[blockIdx.x] = shared[0];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
		__device__ double* buffer = NULL;
	}
}

__device__ double ss_sum(const double* x, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_sum_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N / 2, 1, grid, block);
		grid_block_init = true;
	}
	
	if (!buffer)
		CUDA_ERR_CHECK(cudaMalloc(&buffer, sizeof(double) * grid.x));

	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block, block.x * sizeof(double)>>>(x, buffer));
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	double result = buffer[0];
	for (int i = 1; i < grid.x; i++)
		result += buffer[i];
	
    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

namespace gpu
{
	namespace ss_norm2_kernel
	{
		// computes the 2-norm of x
		// x is a vector of length N
		__global__ void kernel(const double* x, double* result)
		{
			extern __shared__ double shared[];

			// Each block hanldes (2 * blockDim.x) elements of reduction.
			int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

			// Load product of first 2 pairs into shared memory:
			// idx-th and (idx + blockDim.x)-th.
			shared[threadIdx.x] = x[i] * x[i] + x[i + blockDim.x] * x[i + blockDim.x];
	
			__syncthreads();

			// Reduce pairs in shared memory.
			for (int s = blockDim.x / 2; s > 32; s >>= 1)
			{
				if (threadIdx.x < s)
					shared[threadIdx.x] += shared[threadIdx.x + s];
		
				__syncthreads();
			}
	
			// Unroll last 32 iterations of loop.
			// There is no need for synchronizations, since all accesses
			// are within single warp.
			if (threadIdx.x < 32)
			{
				volatile double* vshared = shared;
				vshared[threadIdx.x] += vshared[threadIdx.x + 32];
				vshared[threadIdx.x] += vshared[threadIdx.x + 16];
				vshared[threadIdx.x] += vshared[threadIdx.x +  8];
				vshared[threadIdx.x] += vshared[threadIdx.x +  4];
				vshared[threadIdx.x] += vshared[threadIdx.x +  2];
				vshared[threadIdx.x] += vshared[threadIdx.x +  1];
			}

			// The first thread writes the result.
			if (threadIdx.x == 0)
				result[blockIdx.x] = shared[0];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
		__device__ double* buffer = NULL;
	}
}

__device__ double ss_norm2(const double* x, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_norm2_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N / 2, 1, grid, block);
		grid_block_init = true;
	}
	
	if (!buffer)
		CUDA_ERR_CHECK(cudaMalloc(&buffer, sizeof(double) * grid.x));

	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block, block.x * sizeof(double)>>>(x, buffer));
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	double result = buffer[0];
	for (int i = 1; i < grid.x; i++)
		result += buffer[i];
	
    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return sqrt(result);
}

namespace gpu
{
	namespace ss_fill_kernel
	{
		// sets entries in a vector to value
		// x is a vector of length N
		// value is th
		__global__ void kernel(double* x, const double value, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;

			x[i] = value;
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_fill(double* x, const double value, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_fill_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(x, value, N));
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

namespace gpu
{
	namespace ss_axpy_kernel
	{
		// computes y := alpha*x + y
		// x and y are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double alpha, const double* x, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;
		
			y[i] += alpha * x[i];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_axpy(double* y, const double alpha, const double* x, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_axpy_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, alpha, x, N));

	// record the number of floating point oporations
	flops_blas1 = flops_blas1 + 2 * N;
}

namespace gpu
{
	namespace ss_add_scaled_diff_kernel
	{
		// computes y = x + alpha*(l-r)
		// y, x, l and r are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double* x, const double alpha,
			const double* l, const double* r, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;

			y[i] = x[i] + alpha * (l[i] - r[i]);
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_add_scaled_diff(double* y, const double* x, const double alpha,
	const double* l, const double* r, const int N)
{
	using namespace gpu;
	using namespace ss_add_scaled_diff_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, x, alpha, l, r, N));

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 3 * N;
}

namespace gpu
{
	namespace ss_scaled_diff_kernel
	{
		// computes y = alpha*(l-r)
		// y, l and r are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double alpha,
			const double* l, const double* r, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;

			y[i] = alpha * (l[i] - r[i]);
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_scaled_diff(double* y, const double alpha,
    const double* l, const double* r, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_scaled_diff_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, alpha, l, r, N));

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
}

namespace gpu
{
	namespace ss_scale_kernel
	{
		// computes y := alpha*x
		// alpha is scalar
		// y and x are vectors of length N
		__global__ void kernel(double* y, const double alpha, double* x, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;

			y[i] = alpha * x[i];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_scale(double* y, const double alpha, double* x, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_scale_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, alpha, x, N));

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + N;
}

namespace gpu
{
	namespace ss_lcomb_kernel
	{
		// computes linear combination of two vectors y := alpha*x + beta*z
		// alpha and beta are scalar
		// y, x and z are vectors of length N
		__global__ void kernel(double* y, const double alpha, double* x, const double beta,
			const double* z, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;
			
			y[i] = alpha * x[i] + beta * z[i];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_lcomb(double* y, const double alpha, double* x, const double beta,
    const double* z, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_lcomb_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, alpha, x, beta, z, N));

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 3 * N;
}

namespace gpu
{
	namespace ss_copy_kernel
	{
		// copy one vector into another y := x
		// x and y are vectors of length N
		__global__ void kernel(double* y, const double* x, const int N)
		{
			int i = blockDim.x * blockIdx.x + threadIdx.x;
			if (i >= N) return;

			y[i] = x[i];
		}

		__device__ dim3 grid, block;
		__device__ bool grid_block_init = false;
	}
}

__device__ void ss_copy(double* y, const double* x, const int N)
{
	using namespace gpu;
	using namespace gpu::ss_copy_kernel;

	if (!grid_block_init)
	{
		get_optimal_grid_block_config(kernel, N, 1, grid, block);
		grid_block_init = true;
	}
	CUDA_LAUNCH_ERR_CHECK(kernel<<<grid, block>>>(y, x, N));
}

namespace gpu
{
	// conjugate gradient solver
	// solve the linear system A*x = b for x
	// the matrix A is implicit in the objective function for the diffusion equation
	// the value in x constitute the "first guess" at the solution
	// x(N)
	// ON ENTRY contains the initial guess for the solution
	// ON EXIT  contains the solution
	__device__ void ss_cg(double* x, const double* b, const int maxiters, const double tol, int* success)
	{
		// this is the dimension of the linear system that we are to solve
		int N = options.N;

		if (!cg_initialized)
		{
		    printf("INITIALIZING CG STATE\n");
		    cg_init(N);
		}

		// epslion value use for matrix-vector approximation
		double eps     = 1.e-8;
		double eps_inv = 1. / eps;

		// allocate memory for temporary storage
		ss_fill(Fx,    0.0, N);
		ss_fill(Fxold, 0.0, N);
		ss_copy(xold, x, N);

		// matrix vector multiplication is approximated with
		// A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
		//     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
		// we compute Fxold at startup
		// we have to keep x so that we can compute the F(x+exps*v)
		diffusion(x, Fxold);

		// v = x + epsilon*x
		ss_scale(v, 1.0 + eps, x, N);

		// Fx = F(v)
		diffusion(v, Fx);

		// r = b - A*x
		// where A*x = (Fx-Fxold)/eps
		ss_add_scaled_diff(r, b, -eps_inv, Fx, Fxold, N);

		// p = r
		ss_copy(p, r, N);

		// rold = <r,r>
		double rold = ss_dot(r, r, N), rnew = rold;

		// check for convergence
		*success = 0;
		if (sqrt(rold) < tol)
		{
		    *success = 1;
		    return;
		}

		int iter;
		for (iter = 1; iter <= maxiters; iter++)
		{
		    // Ap = A*p
		    ss_lcomb(v, 1.0, xold, eps, p, N);
		    diffusion(v, Fx);
		    ss_scaled_diff(Ap, eps_inv, Fx, Fxold, N);

		    // alpha = rold / p'*Ap
		    double alpha = rold / ss_dot(p, Ap, N);

		    // x += alpha*p
		    ss_axpy(x, alpha, p, N);

		    // r -= alpha*Ap
		    ss_axpy(r, -alpha, Ap, N);

		    // find new norm
		    rnew = ss_dot(r, r, N);

		    // test for convergence
		    if (sqrt(rnew) < tol)
		    {
		        *success = 1;
		        break;
		    }

		    // p = r + rnew.rold * p
		    ss_lcomb(p, 1.0, r, rnew / rold, p, N);

		    rold = rnew;
		}
		iters_cg += iter;

		if (!*success)
		{
		    printf("ERROR: CG failed to converge after %d iterations\n", maxiters);
		    printf("       achived tol = %E, required tol = %E\n", sqrt(rnew), tol);
		}
	}
}

