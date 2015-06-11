// ******************************************
// implicit time stepping implementation of 2D diffusion problem
// Ben Cumming, CSCS
// *****************************************

// A small benchmark app that solves the 2D fisher equation using second-order
// finite differences.

// Syntax: ./main nx ny nt t
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

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

#ifdef _VISIT_
#include <VisItControlInterface_V2.h>
#include <VisItDataInterface_V2.h>

/* Data Access Function prototypes */
visit_handle SimGetMetaData(void *);
visit_handle SimGetMesh(int, const char *, void *);
visit_handle SimGetVariable(int, const char *, void *);
visit_handle SimGetDomainList(const char *, void *);
void ControlCommandCallback(const char *cmd, const char *args, void *cbdata);
void SlaveProcessCallback(void *cbdata);
int ProcessVisItCommand(void);
void SetupCallbacks(void);
int visit_broadcast_int_callback(int *value, int sender, void *cbdata);
int visit_broadcast_string_callback(char *str, int len, int sender, void *cbdata);
#endif

// ==============================================================================
void write_binary(std::string fname, Field &u, SubDomain &domain, Discretization &options)
{
    MPI_Offset disp = 0;
    MPI_File filehandle;
    MPI_Datatype filetype;

    int result =
        MPI_File_open(
            MPI_COMM_WORLD,
            (char*)fname.c_str(),
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
            MPI_INFO_NULL,
            &filehandle
        );
    assert(result==MPI_SUCCESS);

    int ustart[]  = {domain.startx-1, domain.starty-1};
    int ucount[]  = {domain.nx, domain.ny};
    int dimuids[] = {options.nx, options.ny};

    result = MPI_Type_create_subarray(2, dimuids, ucount, ustart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &filetype);
    assert(result==MPI_SUCCESS);

    result = MPI_Type_commit(&filetype);
    assert(result==MPI_SUCCESS);

    result = MPI_File_set_view(filehandle, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    assert(result==MPI_SUCCESS);

    result = MPI_File_write_all(filehandle, u.data(), domain.N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    assert(result==MPI_SUCCESS);

    result = MPI_Type_free(&filetype);
    assert(result==MPI_SUCCESS);

    result = MPI_File_close(&filehandle);
    assert(result==MPI_SUCCESS);
}

// read command line arguments
static void readcmdline(Discretization& options, int argc, char* argv[])
{
    if (argc<5 || argc>6 ) {
        std::cerr << "Usage: main nx ny nt t\n";
        std::cerr << "  nx  number of gridpoints in x-direction\n";
        std::cerr << "  ny  number of gridpoints in y-direction\n";
        std::cerr << "  nt  number of timesteps\n";
        std::cerr << "  t   total time\n";
        std::cerr << "  v   [optional] turn on verbose output\n";
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

    verbose_output = false;
    if( argc==6 ) {
        verbose_output = (domain.rank==0);
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
int simulate_one_timestep(void)
{
    int max_cg_iters     = 200;
    int max_newton_iters = 50;
    double tolerance     = 1.e-6;
        // set x_new and x_old to be the solution
        ss_copy(x_old, x_new);

        double residual;
        bool converged = false;
        int it;
        for (it=0; it<max_newton_iters; it++)
        {
            // compute residual : requires both x_new and x_old
            diffusion(x_new, b);
            residual = ss_norm2(b);

            // check for convergence
            if (residual < tolerance)
            {
                converged = true;
                break;
            }

            // solve linear system to get -deltax
            bool cg_converged = false;
            ss_cg(deltax, b, max_cg_iters, tolerance, cg_converged);

            // check that the CG solver converged
            if (!cg_converged) break;

            // update solution
            ss_axpy(x_new, -1.0, deltax);
        }
        iters_newton += it+1;

        #ifdef OUTPUT_EVERY_STEP
        std::stringstream str;
        str << "output" << timestep << ".bin"; // get filename for this time step
        write_binary(str.str(), x_old, domain, options); // write binary solution to file
        #endif
/*
        // output some statistics
        if (converged && verbose_output) {
            std::cout << "step " << options.ct
                      << " required " << it
                      << " iterations for residual " << residual
                      << std::endl;
        }*/
        if (!converged) {
            return -1;
        }
      VisItTimeStepChanged();
      VisItUpdatePlots();
    return 0;
}

// ==============================================================================

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(options, argc, argv);

    // initialize MPI
    int i, mpi_rank, mpi_size;
    if( MPI_Init(&argc, &argv) != MPI_SUCCESS ) {
        std::cerr << "unable to initialize MPI :: exitting" << std::endl;
        exit(-1);
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // initialize subdomain
    domain.init(mpi_rank, mpi_size, options);
    //domain.print();

    int nx = domain.nx;
    int ny = domain.ny;
    int N  = domain.N;
    int nt  = options.nt;

    if( domain.rank == 0 ) {
        std::cout << "========================================================================" << std::endl;
        std::cout << "                      Welcome to mini-stencil!" << std::endl;
        std::cout << "version   :: CUDA with MPI : " << domain.size << " MPI ranks" << std::endl;
        std::cout << "mesh      :: " << options.nx << " * " << options.ny << " dx = " << options.dx << std::endl;
        std::cout << "time      :: " << nt << " time steps from 0 .. " << options.nt*options.dt << std::endl;;
        //std::cout << "iteration :: " << "CG "          << max_cg_iters
                                     //<< ", Newton "    << max_newton_iters
                                     //<< ", tolerance " << tolerance << std::endl;;
        std::cout << "========================================================================" << std::endl;
    }

#ifdef _VISIT_
  runMode = false;
  char fname[128];
  sprintf(fname, "/tmp/trace.%03d.txt", domain.rank);
  VisItOpenTraceFile(fname);
  VisItSetBroadcastIntFunction2(visit_broadcast_int_callback, NULL);
  VisItSetBroadcastStringFunction2(visit_broadcast_string_callback, NULL);
  VisItSetParallel(domain.size > 1);
  VisItSetParallelRank(domain.rank);
  VisItSetMPICommunicator((void*)&comm);
  char *env = NULL;
  if(domain.rank == 0)
     env = VisItGetEnvironment();

    /* Pass the environment to all other processors collectively. */
  VisItSetupEnvironment2(env);
  if(env != NULL)
    free(env);

  if(domain.rank == 0)
    VisItInitializeSocketAndDumpSimFile(
            "MiniApp",
            "MiniApp solver for Diffusion Equation",
            "/path/to/where/sim/was/started",
            NULL, NULL, "/users/jfavre/.visit/simulations/miniapp.sim2");
#endif

    // allocate global fields
    x_new.init(nx,ny);
    x_old.init(nx,ny);
    bndN.init(nx,1);
    bndS.init(nx,1);
    bndE.init(ny,1);
    bndW.init(ny,1);
    buffN.init(nx,1);
    buffS.init(nx,1);
    buffE.init(ny,1);
    buffW.init(ny,1);

    b.init(nx,ny);
    deltax.init(nx,ny);

    // set dirichlet boundary conditions to 0 all around
    ss_fill(bndN, 0.);
    ss_fill(bndS, 0.);
    ss_fill(bndE, 0.);
    ss_fill(bndW, 0.);

#ifdef _VISIT_
    coords_x.init(nx,1);
    for (i = 0; i < nx; i++)
      coords_x[i] = options.dx*(i+domain.startx-1);

    coords_y.init(ny,1);
    for (i = 0; i < ny; i++)
      coords_y[i] = options.dx*(i+domain.starty-1);
#endif

    // set the initial condition
    // a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    // no larger than 1/8 of both xdim and ydim
    ss_fill(x_new, 0.);
    double xc = 1.0 / 4.0;
    double yc = (options.ny - 1) * options.dx / 4;
    double radius = std::min(xc, yc) / 2.0;
    for (int j = domain.starty-1; j < domain.endy; j++)
    {
        double y = (j - 1) * options.dx;
        for (int i = domain.startx-1; i < domain.endx; i++)
        {
            double x = (i - 1) * options.dx;
            if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
                x_new(i-domain.startx+1, j-domain.starty+1) = 0.1;
        }
    }

    flops_bc = 0;
    flops_diff = 0;
    flops_blas1 = 0;
    iters_cg = 0;
    iters_newton = 0;

    // start timer
    //double timespent = -omp_get_wtime();
    if(domain.rank == 0) 
        std::cout << "ENTERING LOOP"
                  << std::endl;
    // main timeloop
    int visitstate, blocking;
    for (int timestep = 1; timestep <= nt; timestep++)
    {
#ifdef _VISIT_
    options.ct = timestep;
    blocking = runMode? 0 : 1;

    if(domain.rank == 0)
      visitstate = VisItDetectInput(blocking, -1);
    MPI_Bcast(&visitstate, 1, MPI_INT, 0, comm);
    // Do different things depending on the output from VisItDetectInput.
    switch(visitstate)
      {
      case 0:
        /* There was no input from VisIt, return control to sim. */
        simulate_one_timestep();
      break;
      case 1:
            /* VisIt is trying to connect to sim. */
        if(VisItAttemptToCompleteConnection() == VISIT_OKAY)
          {
          if(domain.rank == 0)
            {
            fprintf(stderr, "VisIt connected\n");
            }
          SetupCallbacks();
          }
        else 
          {
           /* Print the error message */
          char *err = VisItGetLastError();
          fprintf(stderr, "VisIt did not connect: %s\n", err);
          free(err);
          }
      break;
      case 2:
        /* VisIt wants to tell the engine something. */
        if(!ProcessVisItCommand())
          {
          /* Disconnect on an error or closed connection. */
          VisItDisconnect();
          if(domain.rank == 0)
            {
            fprintf(stderr, "VisIt disconnected\n");
            } 
          runMode = true;
          }
      break;
      default:
        fprintf(stderr, "Can't recover from error %d!\n", visitstate);
      break;
      }
#else
    if (simulate_one_timestep())
       {
        std::cerr << "step " << timestep
                  << " ERROR : nonlinear iterations failed to converge" << std::endl;;
        break;
       }
#endif
    }

    // get times
    //timespent += omp_get_wtime();
    unsigned long long flops_total = flops_diff + flops_blas1;

    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

    // binary data
    write_binary("output.bin", x_old, domain, options);

    // metadata
    if( domain.rank==0 ) {
        std::ofstream fid1("output.bov");
        fid1 << "TIME: 0.0" << std::endl;
        fid1 << "DATA_FILE: output.bin" << std::endl;
        fid1 << "DATA_SIZE: " << options.nx << ", " << options.ny << ", 1" << std::endl;;
        fid1 << "DATA_FORMAT: DOUBLE" << std::endl;
        fid1 << "VARIABLE: phi" << std::endl;
        fid1 << "DATA_ENDIAN: LITTLE" << std::endl;
        fid1 << "CENTERING: nodal" << std::endl;
        fid1 << "BRICK_SIZE: 1.0 " << (options.ny-1)*options.dx << " 1.0" << std::endl;

        std::ofstream fid2("output.xmf");
        fid2 << "<?xml version=\"1.0\" ?>" << std::endl;
        fid2 << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
        fid2 << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
        fid2 << "  <Domain>" << std::endl;
        fid2 << "    <Grid Name=\"Jacobi Mesh\" GridType=\"Uniform\">" << std::endl;
        fid2 << "      <Topology TopologyType=\"3DCORECTMESH\" Dimensions=\"1 " << options.nx << " " << options.ny << "\"/>" << std::endl;
        fid2 << "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">" << std::endl;
        fid2 << "         <DataItem Name=\"Origin\" NumberType=\"Float\" Dimensions=\"3\" Format=\"XML\">0. 0. 0.</DataItem>" << std::endl;
        fid2 << "         <DataItem Name=\"Spacing\" NumberType=\"Float\" Dimensions=\"3\" Format=\"XML\">1. 1. 1.</DataItem>" << std::endl;
        fid2 << "      </Geometry>" << std::endl;
        fid2 << "      <Attribute Name=\"Temperature\" Active=\"1\" AttributeType=\"Scalar\" Center=\"Node\">" << std::endl;
        fid2 << "          <DataItem Dimensions=\"1 " << options.nx << " " << options.ny << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\">output.bin</DataItem>" << std::endl;
        fid2 << "      </Attribute>" << std::endl;
        fid2 << "    </Grid>" << std::endl;
        fid2 << "  </Domain>" << std::endl;
        fid2 << "</Xdmf>" << std::endl;
    }

    // print table sumarizing results
    if(domain.rank == 0) {
        std::cout << "--------------------------------------------------------------------------------"
                  << std::endl;
        //std::cout << "simulation took " << timespent << " seconds" << std::endl;
        //std::cout </ int(iters_cg) << " conjugate gradient iterations, at rate of "
                  //<< float(iters_cg)/timespent << " iters/second" << std::endl;
        std::cout << iters_newton << " newton iterations" << std::endl;
        std::cout << "--------------------------------------------------------------------------------"
                  << std::endl;
    }

    if(domain.rank==0)
        std::cout << "Goodbye!" << std::endl;

    MPI_Finalize();

    return 0;
}

