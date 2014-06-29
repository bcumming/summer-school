// linear algebra subroutines
// Ben Cumming @ CSCS

#ifndef LINALG_H
#define LINALG_H
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "data.h"

    extern int cg_initialized;
    extern cl_mem r, Ap, p, Fx, Fxold, v, xold, rold, rnew, scratchBuff,dotProduct; // 1d 

    // initialize temporary storage fields used by the cg solver
    // I do this here so that the fields are persistent between calls
    // to the CG solver. This is useful if we want to avoid malloc/free calls
    // on the device for the OpenACC implementation (feel free to suggest a better
    // method for doing this)
    void cg_init(const int N, cl_context * context,cl_program *program, cl_kernel * kernel,cl_mem b_device, double * eps_inv);


    void ss_cg(const int maxiters, const double tol, int* success, cl_mem bnd_device, cl_mem x_new_device, cl_mem x_old_device, cl_mem b_device, cl_mem deltax_device, cl_command_queue * command_queue,cl_context* context,cl_program *program, cl_kernel *kernel, cl_double dxs, cl_double alpha_device,int nx, int ny);


#endif // LINALG_H

