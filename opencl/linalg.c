// linear algebra subroutines
// Ben Cumming @ CSCS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include <clAmdBlas.h>

#include "linalg.h"
#include "operators.h"
#include "stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int cg_initialized = 0;
cl_mem r = NULL, Ap = NULL, p = NULL;
cl_mem Fx = NULL, Fxold = NULL, v = NULL, xold = NULL,rold= NULL, rnew=NULL, scratchBuff=NULL, dotProduct=NULL; // 1d


// initialize temporary storage fields used by the cg solver
// I do this here so that the fields are persistent between calls
// to the CG solver. This is useful if we want to avoid malloc/free calls
// on the device for the OpenACC implementation (feel free to suggest a better
// method for doing this)
void cg_init(const int N,cl_context * context, cl_program * program, cl_kernel * kernel, cl_mem b_device, double * eps_inv)
{
    cl_int ret;
	Ap = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	r = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	p = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	Fx = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	Fxold = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	v = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	xold = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	rold = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	rnew = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	scratchBuff = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	dotProduct = clCreateBuffer(*context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("CAN'T ALLOCATE DEVICE MEMORY\n");
	
	ret=clSetKernelArg(kernel[2],0,sizeof(cl_mem),(void*)&r);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[2],1,sizeof(cl_mem),(void*)&b_device);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[2],2,sizeof(double),eps_inv);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[2],3,sizeof(cl_mem),(void*)&Fx);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[2],4,sizeof(cl_mem),(void*)&Fxold);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[5],0,sizeof(cl_mem),(void*)&Ap);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[5],2,sizeof(cl_mem),(void*)&Fx);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
	ret=clSetKernelArg(kernel[5],3,sizeof(cl_mem),(void*)&Fxold);
	if (ret!=CL_SUCCESS)
		printf("CAN'T SET KERNEL ARGUMENT\n");
			
    cg_initialized = 1;
}



// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
//x=deltax
void ss_cg(const int maxiters, const double tol, int* success, cl_mem bnd_device, cl_mem x_new_device, cl_mem x_old_device, cl_mem b_device, cl_mem deltax_device, cl_command_queue * command_queue,cl_context * context,cl_program * program, cl_kernel * kernel,cl_double dxs, cl_double alpha_device,int nx, int ny)
{
    cl_event event = NULL;
	cl_int ret;
	char err;
	//struct discretization_t* options = data::options;
	 // epslion value use for matrix-vector approximation
    double eps     = 1.e-8;
    double eps_inv = 1. / eps;
	// this is the dimension of the linear system that we are to solve
    int N = options.N;

    if (!cg_initialized)
    {
        printf("INITIALIZING CG STATE\n");
        cg_init(N, context,program,kernel,b_device,&eps_inv);
    }

   
	// allocate memory for temporary storage
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//ss_fill(Fx,    0.0, N);
    //ss_fill(Fxold, 0.0, N);
    //ss_copy(xold, x, N);
	cl_double pattern=0;
	err=clEnqueueFillBuffer(*command_queue, Fx, &pattern,sizeof(double),0,N*sizeof(double),0,NULL,&event);
	if (err != CL_SUCCESS) {
		printf("clAmdFill() failed with %d\n", err);
		ret = 1;
	}
	else {
		err = clWaitForEvents(1, &event);
	}
		
	err=clEnqueueFillBuffer(*command_queue, Fxold, &pattern,sizeof(double),0,N*sizeof(double),0,NULL,&event);
	if (err != CL_SUCCESS) {
		printf("clAmdFill() failed with %d\n", err);
		ret = 1;
	}
	else {
		err = clWaitForEvents(1, &event);
	}
	
	err=ret=clEnqueueCopyBuffer(*command_queue, deltax_device,xold,0,0,N*sizeof(double), 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("clAmdFill() failed with %d\n", err);
		ret = 1;
	}
	else {
		err = clWaitForEvents(1, &event);
	}
	
	
	
    // matrix vector multiplication is approximated with
    // A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    //     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    // we compute Fxold at startup
    // we have to keep x so that we can compute the F(x+exps*v)
    	
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void*)&deltax_device);
	ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void*)&Fxold);
	ret=clSetKernelArg(kernel[0],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[0],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[0],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[0],5,sizeof(int),&ny);
	ret=clSetKernelArg(kernel[0],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[0],7,sizeof(double),&alpha_device);
	size_t global_item_size1[2]={nx,ny};
	size_t local_item_size1[2]={16,16};
	err= clEnqueueNDRangeKernel(*command_queue, kernel[0], 2, NULL, global_item_size1, local_item_size1, 0 ,NULL, &event);
	
	
	ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void*)&deltax_device);
	ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void*)&Fxold);
	ret=clSetKernelArg(kernel[1],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[1],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[1],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[1],5,sizeof(int),&ny);
	dxs = (1000.0*options.dx*options.dx);
	ret=clSetKernelArg(kernel[1],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[1],7,sizeof(double),&alpha_device);
	size_t global_item_size2[2]={(ny-2),2};
	size_t local_item_size2[2]={256,1};
	err= clEnqueueNDRangeKernel(*command_queue, kernel[1], 2, NULL, global_item_size2, local_item_size2, 0 ,NULL, &event);
	
	
	ret=clSetKernelArg(kernel[6],0,sizeof(cl_mem),(void*)&deltax_device);
	ret=clSetKernelArg(kernel[6],1,sizeof(cl_mem),(void*)&Fxold);
	ret=clSetKernelArg(kernel[6],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[6],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[6],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[6],5,sizeof(int),&ny);
	dxs = (1000.0*options.dx*options.dx);
	ret=clSetKernelArg(kernel[6],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[6],7,sizeof(double),&alpha_device);
	size_t global_item_size3[2]={(nx),2};
	size_t local_item_size3[2]={256,1};
	err= clEnqueueNDRangeKernel(*command_queue, kernel[6], 1, NULL, global_item_size3, local_item_size3, 0 ,NULL, &event);
	
	
	ret=clSetKernelArg(kernel[3],0,sizeof(cl_mem),(void*)&v);
	ret=clSetKernelArg(kernel[3],1,sizeof(cl_mem),(void*)&deltax_device);
	cl_double eps_blas2=1.0+eps;
	ret=clSetKernelArg(kernel[3],2,sizeof(double),&eps_blas2);
	size_t global_item_size=N;
	size_t local_item_size= 256;
	ret= clEnqueueNDRangeKernel(*command_queue, kernel[3], 1, NULL, &global_item_size, &local_item_size, 0 ,NULL, NULL);
	
	
	
	
    // Fx = F(v)
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
	
	ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void*)&v);
	ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void*)&Fx);
	ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void*)&v);
	ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void*)&Fx);
	ret=clSetKernelArg(kernel[6],0,sizeof(cl_mem),(void*)&v);
	ret=clSetKernelArg(kernel[6],1,sizeof(cl_mem),(void*)&Fx);
	//global_item_size=N;
	//local_item_size=256;
	ret= clEnqueueNDRangeKernel(*command_queue, kernel[0], 2, NULL, global_item_size1, local_item_size1, 0 ,NULL, NULL);
	ret= clEnqueueNDRangeKernel(*command_queue, kernel[1], 1, NULL, global_item_size2, local_item_size2, 0 ,NULL, NULL);
	ret= clEnqueueNDRangeKernel(*command_queue, kernel[6], 1, NULL, global_item_size3, local_item_size3, 0 ,NULL, NULL);
	 
	//diffusion(v, Fx);

    // r = b - A*x
    // where A*x = (Fx-Fxold)/eps
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//ss_add_scaled_diff(r, b, -eps_inv, Fx, Fxold, N);
	//merged_blas1();
	
	ret=clSetKernelArg(kernel[2],0,sizeof(cl_mem),(void*)&r);
	ret=clSetKernelArg(kernel[2],1,sizeof(cl_mem),(void*)&b_device);
	global_item_size=N;
	local_item_size= 256;
	cl_double eps_blas1=-eps_inv;
	ret=clSetKernelArg(kernel[2],2,sizeof(cl_double),&eps_blas1);
	ret=clSetKernelArg(kernel[2],3,sizeof(cl_mem),(void*)&Fx);
	ret=clSetKernelArg(kernel[2],4,sizeof(cl_mem),(void*)&Fxold);
	ret=clSetKernelArg(kernel[2],5,sizeof(cl_mem),(void*)&p);
	ret= clEnqueueNDRangeKernel(*command_queue, kernel[2], 1, NULL, &global_item_size, &local_item_size, 0 ,NULL, NULL);
	
    // p = r
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//ss_copy(p, r, N);

    // rold = <r,r>
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//double rold = ss_dot(r, r, N), rnew = rold;

	clAmdBlasDdot(N,rold,0,r,0,1,r,0,1,scratchBuff,1,command_queue,0,NULL,NULL);
	
	
	//ret=clEnqueueCopyBuffer(*command_queue, dotProduct,rold,0,0,sizeof(double), 0, NULL, NULL);
		
	ret=clEnqueueCopyBuffer(*command_queue, rold,rnew,0,0,sizeof(double), 0, NULL, NULL);
	double rold_cpu,rnew_cpu;
	ret=clEnqueueReadBuffer(*command_queue, rold, CL_TRUE, 0 ,sizeof(double), &rold_cpu,0, NULL, NULL);
	
    // check for convergence
    *success = 0;
    if (sqrt(rold_cpu) < tol)
    {
        *success = 1;
        return;
    }

    int iter;
    for (iter = 1; iter <= maxiters; iter++)
    {
        // Ap = A*p
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
		ret=clSetKernelArg(kernel[4],0,sizeof(cl_mem),(void*)&v);
		ret=clSetKernelArg(kernel[4],1,sizeof(cl_mem),(void*)&xold);
		cl_double alpha_lcomb=1.0;
		ret=clSetKernelArg(kernel[4],2,sizeof(double),&alpha_lcomb);
		cl_double beta_lcomb=eps;
		ret=clSetKernelArg(kernel[4],3,sizeof(double),&beta_lcomb);
		ret=clSetKernelArg(kernel[4],4,sizeof(double),&p);
		global_item_size=N;
		local_item_size= 256;
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[4], 1, NULL, &global_item_size, &local_item_size, 0 ,NULL, NULL);
		if (ret != CL_SUCCESS) {
		printf("clAmdFill() failed with %d\n", ret);
		ret = 1;
		}
	
		ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void*)&v);
		ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void*)&Fx);
		ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void*)&v);
		ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void*)&Fx);
		ret=clSetKernelArg(kernel[6],0,sizeof(cl_mem),(void*)&v);
		ret=clSetKernelArg(kernel[6],1,sizeof(cl_mem),(void*)&Fx);
		//global_item_size=N;
		//local_item_size=256;
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[0], 2, NULL, global_item_size1, local_item_size1, 0 ,NULL, NULL);
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[1], 1, NULL, global_item_size2, local_item_size2, 0 ,NULL, NULL);
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[6], 1, NULL, global_item_size3, local_item_size3, 0 ,NULL, NULL);
    
		
        global_item_size=N;
		local_item_size= 256;
		cl_double eps_inv_blas=eps_inv;
		ret=clSetKernelArg(kernel[5],1,sizeof(cl_double),&eps_inv_blas);
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[5], 1, NULL, &global_item_size, &local_item_size, 0 ,NULL, NULL);
		
        // alpha = rold / p'*Ap
		
		        
		clAmdBlasDdot(N,dotProduct,0,p,0,1,Ap,0,1,scratchBuff,1,command_queue,0,NULL,NULL);
		cl_double ss_dot_res;
		ret=clEnqueueReadBuffer(*command_queue, dotProduct, CL_TRUE, 0 ,sizeof(double), &ss_dot_res,0, NULL, NULL);
	
		
		cl_double cl_alpha = rold_cpu / ss_dot_res;
		
        // x += alpha*p
        cl_double daxpy_alpha = cl_alpha;
		clAmdBlasDaxpy(N, daxpy_alpha, p,0, 1, deltax_device, 0, 1, 1, command_queue,0, NULL, NULL);
		
			
        // r -= alpha*Ap
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//ss_axpy(r, -alpha, Ap, N);
		
		daxpy_alpha = -cl_alpha;
		clAmdBlasDaxpy(N, daxpy_alpha, Ap,0, 1, r, 0, 1, 1, command_queue,0, NULL, NULL);
		
        // find new norm
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		clAmdBlasDdot(N,dotProduct,0,r,0,1,r,0,1,scratchBuff,1,command_queue,0,NULL,NULL);
		ret=clEnqueueReadBuffer(*command_queue, dotProduct, CL_TRUE, 0 ,sizeof(double), &rnew_cpu,0, NULL, NULL);
		ret=clEnqueueCopyBuffer(*command_queue, dotProduct,rnew,0,0,sizeof(double), 0, NULL, NULL);
	

        // test for convergence
        if (sqrt(rnew_cpu) < tol)
        {
            *success = 1;
            break;
        }

        // p = r + rnew.rold * p
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		ret=clSetKernelArg(kernel[4],0,sizeof(cl_mem),(void*)&p);
		ret=clSetKernelArg(kernel[4],1,sizeof(cl_mem),(void*)&r);
		alpha_lcomb=1.0;
		ret=clSetKernelArg(kernel[4],2,sizeof(double),&alpha_lcomb);
		beta_lcomb=rnew_cpu/rold_cpu;
		ret=clSetKernelArg(kernel[4],3,sizeof(double),&beta_lcomb);
		ret=clSetKernelArg(kernel[4],4,sizeof(double),&p);
		global_item_size=N;
		local_item_size= 256;
		ret= clEnqueueNDRangeKernel(*command_queue, kernel[4], 1, NULL, &global_item_size, &local_item_size, 0 ,NULL, NULL);
    
		ret=clEnqueueCopyBuffer(*command_queue, rnew,rold,0,0,sizeof(double), 0, NULL, NULL);
	    rold_cpu=rnew_cpu;
		//rold = rnew;
    }
    iters_cg += iter;

    if (!*success)
    {
        fprintf(stderr, "ERROR: CG failed to converge after %d iterations\n", maxiters);
        fprintf(stderr, "       achived tol = %E, required tol = %E\n", sqrt(rnew_cpu), tol);
    }
}

