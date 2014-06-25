// linear algebra subroutines
// Ben Cumming @ CSCS

#ifndef LINALG_H
#define LINALG_H

#include "data.h"

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors of length N
__device__ double ss_dot(const double* x, const double* y, const int N);

// computes the sum of x elements
// x is a vector of length N
__device__ double ss_sum(const double* x, const int N);

// computes the 2-norm of x
// x is a vector of length N
__device__ double ss_norm2(const double* x, const int N);

// sets entries in a vector to value
// x is a vector of length N
// value is th
__device__ void ss_fill(double* x, const double value, const int N);

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors of length N
// alpha is a scalar
__device__ void ss_axpy(double* y, const double alpha, const double* x, const int N);

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
__device__ void ss_add_scaled_diff(double* y, const double* x, const double alpha,
    const double* l, const double* r, const int N);

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
__device__ void ss_scaled_diff(double* y, const double alpha,
    const double* l, const double* r, const int N);

// computes y := alpha*x
// alpha is scalar
// y and x are vectors of length N
__device__ void ss_scale(double* y, const double alpha, double* x, const int N);

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors of length N
__device__ void ss_lcomb(double* y, const double alpha, double* x, const double beta,
    const double* z, const int N);

// copy one vector into another y := x
// x and y are vectors of length N
__device__ void ss_copy(double* y, const double* x, const int N);

namespace gpu
{
	namespace ss_dot_kernel
	{
		// computes the inner product of x and y
		// x and y are vectors of length N
		__global__ void kernel(const double* x, const double* y, double* result);
		
		extern __constant__ config_t config;
	}

	namespace ss_sum_kernel
	{
		// computes the sum of x elements
		// x is a vector of length N
		__global__ void kernel(const double* x, double* result);

		extern __constant__ config_t config;
	}

	namespace ss_norm2_kernel
	{
		// computes the 2-norm of x
		// x is a vector of length N
		__global__ void kernel(const double* x, double* result);

		extern __constant__ config_t config;
	}

	namespace ss_fill_kernel
	{
		// sets entries in a vector to value
		// x is a vector of length N
		// value is th
		__global__ void kernel(double* x, const double value, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_axpy_kernel
	{
		// computes y := alpha*x + y
		// x and y are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double alpha, const double* x, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_add_scaled_diff_kernel
	{
		// computes y = x + alpha*(l-r)
		// y, x, l and r are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double* x, const double alpha,
			const double* l, const double* r, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_scaled_diff_kernel
	{
		// computes y = alpha*(l-r)
		// y, l and r are vectors of length N
		// alpha is a scalar
		__global__ void kernel(double* y, const double alpha,
			const double* l, const double* r, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_scale_kernel
	{
		// computes y := alpha*x
		// alpha is scalar
		// y and x are vectors of length N
		__global__ void kernel(double* y, const double alpha, double* x, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_lcomb_kernel
	{
		// computes linear combination of two vectors y := alpha*x + beta*z
		// alpha and beta are scalar
		// y, x and z are vectors of length N
		__global__ void kernel(double* y, const double alpha, double* x, const double beta,
			const double* z, const int N);

		extern __constant__ config_t config;
	}

	namespace ss_copy_kernel
	{
		// copy one vector into another y := x
		// x and y are vectors of length N
		__global__ void kernel(double* y, const double* x, const int N);

		extern __constant__ config_t config;
	}

	// conjugate gradient solver
	// solve the linear system A*x = b for x
	// the matrix A is implicit in the objective function for the diffusion equation
	// the value in x constitute the "first guess" at the solution
	// x(N)
	// ON ENTRY contains the initial guess for the solution
	// ON EXIT  contains the solution
	__device__ void ss_cg(double* x, const double* b, const int maxiters, const double tol, int* success);
}

#endif // LINALG_H

