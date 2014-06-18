// ******************************************
// implicit time stepping implementation of 2D diffusion problem
// Ben Cumming, CSCS
// *****************************************

// A small benchmark app that solves the 2D fisher equation using second-order
// finite differences.

// Syntax: ./main nx ny nt t
#include <algorithm>
#include <iostream>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <omp.h>
#include <mpi.h>

#include "data.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

using namespace data;
using namespace linalg;
using namespace operators;
using namespace stats;

// ==============================================================================

// read command line arguments
static void readcmdline(Discretization& options, int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: main nx ny nt t\n";
        std::cerr << "  nx  number of gridpoints in x-direction\n";
        std::cerr << "  ny  number of gridpoints in y-direction\n";
        std::cerr << "  nt  number of timesteps\n";
        std::cerr << "  t   total time\n";
        exit(1);
    }

    // read nx
    options.nx = atoi(argv[1]);
    if (options.nx < 1) {
        std::cerr << "nx must be positive integer\n";
        exit(-1);
    }

    // read ny
    options.ny = atoi(argv[2]);
    if (options.ny < 1) {
        std::cerr << "ny must be positive integer\n";
        exit(-1);
    }

    // read nt
    options.nt = atoi(argv[3]);
    if (options.nt < 1) {
        std::cerr << "nt must be positive integer\n";
        exit(-1);
    }

    // read total time
    double t = atof(argv[4]);
    if (t < 0) {
        std::cerr << "t must be positive real value\n";
        exit(-1);
    }

    // compute timestep size
    options.dt = t / options.nt;

    // compute the distance between grid points
    // assume that x dimension has length 1.0
    options.dx = 1. / (options.nx - 1);

    // set alpha, assume diffusion coefficient D is 1
    options.alpha = (options.dx * options.dx) / (1. * options.dt);
}

// ==============================================================================

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(options, argc, argv);

    // initialize MPI
    int mpi_rank, mpi_size;
    if( MPI_Init(&argc, &argv) != MPI_SUCCESS ) {
        std::cerr << "unable to initialize MPI :: exitting" << std::endl;
        exit(-1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // initialize subdomain
    domain.init(mpi_rank, mpi_size, options);
    domain.print();

    int nx = domain.nx;
    int ny = domain.ny;
    int N  = domain.N;

    int nt  = options.nt;

    if( domain.rank == 0 ) {
        std::cout << "========================================================================" << std::endl;
        std::cout << "                      Welcome to mini-stencil!" << std::endl;
        std::cout << "version :: C++ with MPI : " << domain.size << " MPI ranks" << std::endl;
        std::cout << "mesh    :: " << options.nx << " * " << options.ny << " dx = " << options.dx << std::endl;
        std::cout << "time    :: " << nt << " time steps from 0 .. " << options.nt*options.dt << std::endl;;
        std::cout << "========================================================================" << std::endl;
    }

    // allocate global fields
    x_new = new double[nx * ny];
    x_old = new double[nx * ny];
    bndN = new double[nx];
    bndS = new double[nx];
    bndE = new double[ny];
    bndW = new double[ny];
    buffN = new double[nx];
    buffS = new double[nx];
    buffE = new double[ny];
    buffW = new double[ny];

    double* b = new double[N];
    double* deltax = new double[N];

    // set dirichlet boundary conditions to 0 all around
    ss_fill(bndN, 0., nx);
    ss_fill(bndS, 0., nx);
    ss_fill(bndE, 0., ny);
    ss_fill(bndW, 0., ny);

    // set the initial condition
    // a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    // no larger than 1/8 of both xdim and ydim
    ss_fill(x_new, 0., nx*ny);
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
                x_new[i+nx*j] = 0.1;
        }
    }

    double time_in_bcs = 0.0;
    double time_in_diff = 0.0;
    flops_bc = 0;
    flops_diff = 0;
    flops_blas1 = 0;
    verbose_output = false;
    iters_cg = 0;
    iters_newton = 0;

    Field x_new_twod(x_new, nx, ny);
    Field b_twod(b, nx, ny);

    // start timer
    double timespent = -omp_get_wtime();

    // main timeloop
    double alpha = options.alpha;
    double tolerance = 1.e-6;
    for (int timestep = 1; timestep <= nt; timestep++)
    {
        // set x_new and x_old to be the solution
        ss_copy(x_old, x_new, N);

        double residual;
        bool converged = false;
        int it;
        for (it=0; it<50; it++)
        {
            // compute residual : requires both x_new and x_old
            //diffusion(x_new, b);
            diffusion(x_new_twod, b_twod);
            residual = ss_norm2(b, N);

            // check for convergence
            if (residual < tolerance)
            {
                converged = true;
                break;
            }

            // solve linear system to get -deltax
            bool cg_converged = false;
            ss_cg(deltax, b, 200, tolerance, cg_converged);

            // check that the CG solver converged
            if (!cg_converged) break;

            // update solution
            ss_axpy(x_new, -1.0, deltax, N);
        }
        iters_newton += it+1;

        // output some statistics
        //if (converged && verbose_output)
        if (converged && verbose_output) {
            std::cout << "step " << timestep
                      << " required " << it
                      << " iterations for residual " << residual
                      << std::endl;
        }
        if (!converged) {
            std::cerr << "step " << timestep
                      << " ERROR : nonlinear iterations failed to converge" << std::endl;;
            break;
        }
    }

    // get times
    timespent += omp_get_wtime();
    unsigned long long flops_total = flops_diff + flops_blas1;

    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

    if( domain.rank==0 ) {
        // binary data
        {
            FILE* output = fopen("output.bin", "w");
            fwrite(x_new, sizeof(double), nx * ny, output);
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
            fprintf(output, "BRICK_SIZE: 1.0 %f 1.0\n", (ny - 1) * options.dx);
            fclose(output);
        }
    }

    // print table sumarizing results
    if(domain.rank == 0) {
        std::cout << "--------------------------------------------------------------------------------"
                  << std::endl;
        std::cout << "simulation took " << timespent << " seconds" << std::endl;
        std::cout << int(iters_cg) << " conjugate gradient iterations, at rate of "
                  << float(iters_newton/timespent) << " iters/second" << std::endl;
        std::cout << iters_newton << " newton iterations" << std::endl;
        std::cout << "--------------------------------------------------------------------------------"
                  << std::endl;
    }

    // deallocate global fields
    delete[] x_new;
    delete[] x_old;
    delete[] bndN;
    delete[] bndS;
    delete[] bndE;
    delete[] bndW;
    delete[] buffN;
    delete[] buffS;
    delete[] buffE;
    delete[] buffW;

    if(domain.rank==0)
        std::cout << "Goodbye!" << std::endl;

    MPI_Finalize();

    return 0;
}

