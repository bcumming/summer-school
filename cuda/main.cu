// ******************************************
// implicit time stepping implementation of 2D diffusion problem
// Ben Cumming, CSCS
// C version by Gilles Fourestey, CSCS
// *****************************************

// A small benchmark app that solves the 2D fisher equation using second-order
// finite differences.

// Syntax: ./main nx ny nt t

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "data.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

// ==============================================================================

// read command line arguments
static void readcmdline(struct discretization_t* options, int argc, char* argv[])
{
    if (argc != 5)
    {
        printf("Usage: main nx ny nt t\n");
        printf("  nx  number of gridpoints in x-direction\n");
        printf("  ny  number of gridpoints in y-direction\n");
        printf("  nt  number of timesteps\n");
        printf("  t   total time\n");
        exit(1);
    }

    // read nx
    options->nx = atoi(argv[1]);
    if (options->nx < 1)
    {
        fprintf(stderr, "nx must be positive integer\n");
        exit(-1);
    }

    // read ny
    options->ny = atoi(argv[2]);
    if (options->ny < 1)
    {
        fprintf(stderr, "ny must be positive integer\n");
        exit(-1);
    }

    // read nt
    options->nt = atoi(argv[3]);
    if (options->nt < 1)
    {
        fprintf(stderr, "nt must be positive integer\n");
        exit(-1);
    }
    
    // read total time
    double t = atof(argv[4]);
    if (t < 0)
    {
        fprintf(stderr, "t must be positive real value\n");
        exit(-1);
    }

    // store the parameters
    options->N = options->nx*options->ny;

    // compute timestep size
    options->dt = t / options->nt;
    
    // compute the distance between grid points
    // assume that x dimension has length 1.0
    options->dx = 1./(options->nx - 1);
    
    // set alpha, assume diffusion coefficient D is 1
    options->alpha = (options->dx*options->dx) / (1.*options->dt);
}

// ==============================================================================

namespace gpu
{
	__device__ double residual;
	__device__ int cg_converged;

	__global__ void main(double* x_new)
	{
		using namespace gpu;

		int N  = options.N;
		int nt = options.nt;
	
		// main timeloop
		double tolerance = 1.e-6;
		int timestep;
		for (timestep = 1; timestep <= nt; timestep++)
		{
		    // set x_new and x_old to be the solution
		    ss_copy(x_old, x_new, N);

		    double residual;
		    int    converged = 0;
		    int    it = 1;
		    for ( ; it <= 50; it++)
		    {
		        // compute residual : requires both x_new and x_old
		        diffusion(x_new, b);
		        residual = ss_norm2(b, N);

		        // check for convergence
		        if (residual < tolerance)
		        {
		            converged = 1;
		            break;
		        }

		        // solve linear system to get -deltax
		        cg_converged = 0;
		        ss_cg(deltax, b, 200, tolerance, &cg_converged);

		        // check that the CG solver converged
		        if (!cg_converged) break;

		        // update solution
		        ss_axpy(x_new, -1.0, deltax, N);

		        // print control sum of x_new
		        if (timestep % 50 == 0)
		        {
		        	double sum = ss_sum(x_new, N);
		            printf("sum = %f\n", sum);
				}
		    }
		    iters_newton += it;

		    // output some statistics
		    if (converged && verbose_output)
		        printf("step %d required %d iterations for residual %E\n", timestep, it, residual);
		    if (!converged)
		    {
		        printf("step %d ERROR : nonlinear iterations failed to converge\n", timestep);
		        break;
		    }
		}
	}
}

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(&cpu::options, argc, argv);
    CUDA_ERR_CHECK(cudaMemcpyToSymbol(
    	gpu::options, &cpu::options, sizeof(struct discretization_t)));

	int nx = cpu::options.nx;
	int ny = cpu::options.ny;
	int N  = cpu::options.N;
	int nt = cpu::options.nt;

    printf("========================================================================\n");
    printf("                      Welcome to mini-stencil!\n");
    printf("mesh :: %d * %d, dx = %f\n", nx, ny, cpu::options.dx);
    printf("time :: %d, time steps from 0 .. %f\n", nt, nt * cpu::options.dt);
    printf("========================================================================\n");

    // allocate global fields
    double* cpu_x_new  = (double*)malloc(sizeof(double) * nx * ny);
    {
    	using namespace cpu;

		x_old  = (double*) malloc(sizeof(double) * nx * ny); 
		bndN   = (double*) malloc(sizeof(double) * nx);
		bndS   = (double*) malloc(sizeof(double) * nx); 
		bndE   = (double*) malloc(sizeof(double) * ny); 
		bndW   = (double*) malloc(sizeof(double) * ny); 
	    b      = (double*) malloc(sizeof(double) * N);
	    deltax = (double*) malloc(sizeof(double) * N);

		// set dirichlet boundary conditions to 0 all around
		memset(bndN,   0, sizeof(double) * nx);
		memset(bndS,   0, sizeof(double) * nx);
		memset(bndE,   0, sizeof(double) * ny);
		memset(bndW,   0, sizeof(double) * ny);
		memset(b,      0, sizeof(double) * N);
		memset(deltax, 0, sizeof(double) * N);

		// set the initial condition
		// a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
		// no larger than 1/8 of both xdim and ydim
		memset(cpu_x_new, 0, sizeof(double) * nx * ny);
		double xc = 1.0 / 4.0;
		double yc = (ny - 1) * options.dx / 4;
		double radius = fmin(xc, yc) / 2.0;
		for (int j = 0; j < ny; j++)
		{
		    double y = (j - 1) * options.dx;
		    for (int i = 0; i < nx; i++)
		    {
		        double x = (i - 1) * options.dx;
		        if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
		            cpu_x_new[i + j * nx] = 0.1;
		    }
		}
	}
   	
	cudaMallocDevice(x_old,  sizeof(double) * nx * ny);
	cudaMallocDevice(bndN,   sizeof(double) * nx);
	cudaMallocDevice(bndS,   sizeof(double) * nx);
	cudaMallocDevice(bndE,   sizeof(double) * ny);
	cudaMallocDevice(bndW,   sizeof(double) * ny);
	cudaMallocDevice(b,      sizeof(double) * N);
	cudaMallocDevice(deltax, sizeof(double) * N);

	{
		cudaDeviceProp props;
    	CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, 0));
		CUDA_ERR_CHECK(cudaMemcpyToSymbol(gpu::props, &props,
			sizeof(cudaDeviceProp)));
	}
	
	// copy initial solution to GPU
	double* gpu_x_new;
	CUDA_ERR_CHECK(cudaMalloc(&gpu_x_new, sizeof(double) * nx * ny));
	CUDA_ERR_CHECK(cudaMemcpy(gpu_x_new, cpu_x_new, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));

    // start timer
    double timespent = -omp_get_wtime();
    
    gpu::main<<<1, 1>>>(gpu_x_new);
    
    CUDA_ERR_CHECK(cudaMemcpy(cpu_x_new, gpu_x_new, sizeof(double) * N, cudaMemcpyDeviceToHost));

    // get times
    timespent += omp_get_wtime();
    unsigned long long flops_total = gpu::get_value(gpu::flops_diff) + gpu::get_value(gpu::flops_blas1);

	using namespace cpu;

    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

    // binary data
    {
        FILE* output = fopen("output.bin", "w");
        fwrite(cpu_x_new, sizeof(double), nx * ny, output);
        fclose(output);
    }

    // metadata
    {
        FILE* output = fopen("output.bov", "wb");
        fprintf(output, "TIME: 0.0\n");
        fprintf(output, "DATA_FILE: output.bin\n");
        fprintf(output, "DATA_SIZE: %d, %d, 1\n", nx, ny);
        fprintf(output, "DATA_FORMAT: DOUBLE\n");
        fprintf(output, "VARIABLE: phi\n");
        fprintf(output, "DATA_ENDIAN: LITTLE\n");
        fprintf(output, "CENTERING: nodal\n");
        //fprintf(output, "BYTE_OFFSET: 4\n");
        fprintf(output, "BRICK_SIZE: 1.0 %f 1.0\n", (ny - 1) * options.dx);
        fclose(output);
    }

    // print table sumarizing results
    printf("--------------------------------------------------------------------------------\n");
    printf("simulation took %f seconds (%f GFLOP/s)\n", timespent, flops_total / 1e9 / timespent);
    printf("%u conjugate gradient iterations\n", gpu::get_value(gpu::iters_cg));
    printf("%u newton iterations\n", gpu::get_value(gpu::iters_newton));
    printf("--------------------------------------------------------------------------------\n");

    // deallocate global fields
    CUDA_ERR_CHECK(cudaFree(gpu_x_new));
    free(cpu_x_new);
    free(x_old);
    free(bndN);
    free(bndS);
    free(bndE);
    free(bndW);

    printf("Goodbye!\n");

    return 0;
}

