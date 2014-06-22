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
		CUDA_ERR_CHECK(cudaMalloc(&Ap,    sizeof(double) * N));
		CUDA_ERR_CHECK(cudaMalloc(&r,     sizeof(double) * N)); 
		CUDA_ERR_CHECK(cudaMalloc(&p,     sizeof(double) * N));
		CUDA_ERR_CHECK(cudaMalloc(&Fx,    sizeof(double) * N));
		CUDA_ERR_CHECK(cudaMalloc(&Fxold, sizeof(double) * N));
		CUDA_ERR_CHECK(cudaMalloc(&v,     sizeof(double) * N));
		CUDA_ERR_CHECK(cudaMalloc(&xold,  sizeof(double) * N));

		cg_initialized = 1;
	}
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
__device__ double ss_dot(const double* x, const double* y, const int N)
{
	using namespace gpu;

    double result = 0;
	int i;
    for (i = 0; i < N; i++)
        result += x[i] * y[i];

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

// computes the 2-norm of x
// x is a vector on length N
__device__ double ss_norm2(const double* x, const int N)
{
	using namespace gpu;

    double result = 0;
	int i;
    for (i = 0; i < N; i++)
        result += x[i] * x[i];

    result = sqrt(result);

    // record the number of floating point oporations
    flops_blas1 = flops_blas1 + 2 * N;
    
    return result;
}

namespace gpu
{
	namespace ss_fill_kernel
	{
		// sets entries in a vector to value
		// x is a vector on length N
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
		// x and y are vectors on length N
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
		// y and x are vectors on length n
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
		// y, x and z are vectors on length n
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

// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
__global__ void ss_cg(double* x, const double* b, const int maxiters, const double tol, int* success)
{
    using namespace gpu;

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
    CUDA_LAUNCH_ERR_CHECK(diffusion<<<1, 1>>>(x, Fxold));
    CUDA_ERR_CHECK(cudaDeviceSynchronize());

    // v = x + epsilon*x
    ss_scale(v, 1.0 + eps, x, N);

    // Fx = F(v)
    CUDA_LAUNCH_ERR_CHECK(diffusion<<<1, 1>>>(v, Fx));
    CUDA_ERR_CHECK(cudaDeviceSynchronize());

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
        CUDA_LAUNCH_ERR_CHECK(diffusion<<<1, 1>>>(v, Fx));
        CUDA_ERR_CHECK(cudaDeviceSynchronize());
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

