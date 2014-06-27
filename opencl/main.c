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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include <clAmdBlas.h>

#include "data.h"
#include "linalg.h"
#include "operators.h"
#include "stats.h"

#define MAX_SOURCE_SIZE (0x1000000)

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

int main(int argc, char* argv[])
{
    // read command line arguments
    readcmdline(&options, argc, argv);
    int nx = options.nx;
    int ny = options.ny;
    int N  = options.N;
    int nt = options.nt;

    printf("========================================================================\n");
    printf("                      Welcome to mini-stencil!\n");
    printf("mesh :: %d * %d, dx = %f\n", nx, ny, options.dx);
    printf("time :: %d, time steps from 0 .. %f\n", nt, options.nt * options.dt);
    printf("========================================================================\n");

    // allocate global fields
    x_new = (double*) malloc(sizeof(double)*nx*ny);
    x_old = (double*) malloc(sizeof(double)*nx*ny); 
    bndN  = (double*) malloc(sizeof(double)*nx);
    bndS  = (double*) malloc(sizeof(double)*nx); 
    bndE  = (double*) malloc(sizeof(double)*ny); 
    bndW  = (double*) malloc(sizeof(double)*ny); 
	
    double* b      = (double*) malloc(N*sizeof(double));
    double* deltax = (double*) malloc(N*sizeof(double));

	FILE *fp;
	char *source_str_diffusion_center,*source_str_diffusion_east_west,*source_str_diffusion_north_south_corners,*source_str_merged_blas1, *source_str_merged_blas2,*source_str_merged_blas3,*source_str_merged_blas4;
	size_t source_size[8];
	
	fp= fopen("operators.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_diffusion_center=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[0]=fread(source_str_diffusion_center,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("operators1.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_diffusion_east_west=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[1]=fread(source_str_diffusion_east_west,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("merged_blas1.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_merged_blas1=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[2]=fread(source_str_merged_blas1,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("merged_blas2.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_merged_blas2=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[3]=fread(source_str_merged_blas2,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("merged_blas3.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_merged_blas3=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[4]=fread(source_str_merged_blas3,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("merged_blas4.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_merged_blas4=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[5]=fread(source_str_merged_blas4,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	fp= fopen("operators2.cl","r");
	if (!fp){
		printf("Failed to load kernel/ \n");
		exit(1);
	}
	source_str_diffusion_north_south_corners=(char*)malloc(MAX_SOURCE_SIZE);
	source_size[6]=fread(source_str_diffusion_north_south_corners,1,MAX_SOURCE_SIZE,fp);
	fclose(fp);
	
	
	
	cl_platform_id platform_id=NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs (1,&platform_id, & ret_num_platforms);
	ret = clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (ret!=CL_SUCCESS)
		printf("NO DEVICES");
		
	cl_context context = clCreateContext (NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("NO CONTEXT");
	cl_command_queue command_queue=clCreateCommandQueue (context, device_id, 0 , &ret);
	if (ret!=CL_SUCCESS)
		printf("NO QUEUE");
	
	//initializing clblas
	cl_int err1 = clAmdBlasSetup();
    if (err1 != CL_SUCCESS) {
        printf("clAmdBlasSetup() failed with %d\n", err1);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return 1;
    }
	
	cl_event event = NULL;
	
	cl_mem x_new_device = clCreateBuffer(context, CL_MEM_READ_WRITE, nx*ny*sizeof(double), NULL, &ret);
	cl_mem x_old_device = clCreateBuffer(context, CL_MEM_READ_WRITE, nx*ny*sizeof(double), NULL, &ret);
	cl_mem b_device = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	cl_mem deltax_device = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(double), NULL, &ret);
	cl_mem residual_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, &ret);
	cl_mem scratchBuff_nrm2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*N*sizeof(double), NULL, &ret);
	cl_double residual;
	
	cl_mem bnd_device = clCreateBuffer(context, CL_MEM_READ_WRITE, (2*nx+2*ny)*sizeof(double), NULL, &ret);
	if (ret!=CL_SUCCESS)
		printf("NO BUFFERS");
		
    // set dirichlet boundary conditions to 0 all around
    memset(bndN, 0, sizeof(double) * nx);
    memset(bndS, 0, sizeof(double) * nx);
    memset(bndE, 0, sizeof(double) * ny);
    memset(bndW, 0, sizeof(double) * ny);

	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, 0 ,nx*sizeof(double), bndN,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, nx*sizeof(double) ,nx*sizeof(double), bndS,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, 2*nx*sizeof(double),ny*sizeof(double), bndW,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, 2*nx*sizeof(double)+ny*sizeof(double) ,ny*sizeof(double), bndE,0, NULL, NULL);
	if (ret!=CL_SUCCESS)
		printf("NO COPYING");
	
	//TODO::clEnqueueFillBuffer(command_queue,bndN_device,,0,nx*sizeof(double)NULL, NULL) - only for OpenCL 1.2
	
    // set the initial condition
    // a circle of concentration 0.1 centred at (xdim/4, ydim/4) with radius
    // no larger than 1/8 of both xdim and ydim
    memset(x_new, 0, sizeof(double) * nx * ny);
    double xc = 1.0 / 4.0;
    double yc = (ny - 1) * options.dx / 4;
    double radius = fmin(xc, yc) / 2.0;
    int i,j;
    //
    for (j = 0; j < ny; j++)
    {
        double y = (j - 1) * options.dx;
        for (i = 0; i < nx; i++)
        {
            double x = (i - 1) * options.dx;
            if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
                //((double(*)[nx])x_new)[j][i] = 0.1;
                x_new[i+j*nx] = 0.1;
        }
    }

	ret=clEnqueueWriteBuffer(command_queue, x_new_device, CL_TRUE, 0 ,nx*ny*sizeof(double), x_new,0, NULL, NULL);
	if (ret!=CL_SUCCESS)
		printf("NO COPYING");
	
		
    double time_in_bcs = 0.0;
    double time_in_diff = 0.0;
    flops_bc = 0;
    flops_diff = 0;
    flops_blas1 = 0;
    verbose_output = 0;
    iters_cg = 0;
    iters_newton = 0;

    
    // main timeloop
    double alpha = options.alpha;
    cl_double alpha_device=alpha;
	double tolerance = 1.e-6;
    int timestep;
	
	cl_program program[8];
	program[0]=clCreateProgramWithSource(context,1, (const char **)&source_str_diffusion_center, (const size_t*)&source_size[0], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[1]=clCreateProgramWithSource(context,1, (const char **)&source_str_diffusion_east_west, (const size_t*)&source_size[1], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[2]=clCreateProgramWithSource(context,1, (const char **)&source_str_merged_blas1, (const size_t*)&source_size[2], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[3]=clCreateProgramWithSource(context,1, (const char **)&source_str_merged_blas2, (const size_t*)&source_size[3], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[4]=clCreateProgramWithSource(context,1, (const char **)&source_str_merged_blas3, (const size_t*)&source_size[4], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[5]=clCreateProgramWithSource(context,1, (const char **)&source_str_merged_blas4, (const size_t*)&source_size[5], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	program[6]=clCreateProgramWithSource(context,1, (const char **)&source_str_diffusion_north_south_corners, (const size_t*)&source_size[6], &ret);
	if (ret!=CL_SUCCESS)
		printf("NO PROGRAM");
	
	const char options_cl[] ="";
	ret=clBuildProgram(program[0],1, &device_id, options_cl, NULL, NULL);
	char err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[1],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[2],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[3],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[4],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[5],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	ret=clBuildProgram(program[6],1, &device_id, options_cl, NULL, NULL);
	err=ret;
	if (ret!=CL_SUCCESS)
		printf("%d",err);
	
	cl_kernel kernel[8];
	kernel[0]= clCreateKernel(program[0], "cl_diffusion_center", &ret);
	if (ret!=CL_SUCCESS)
		printf("NO KERNEL 0");
	kernel[1]= clCreateKernel(program[1], "cl_diffusion_east_west", &ret);
	if (ret!=CL_SUCCESS)
		printf("NO KERNEL 1");
	kernel[2]= clCreateKernel(program[2], "cl_merged_blas1", &ret);
	if (ret!=CL_SUCCESS)
		printf("NO KERNEL 2");
	kernel[3]= clCreateKernel(program[3], "cl_merged_blas2", &ret);
		if (ret!=CL_SUCCESS)
		printf("NO KERNEL 3");
	kernel[4]= clCreateKernel(program[4], "cl_merged_blas3", &ret);
		if (ret!=CL_SUCCESS)
		printf("NO KERNEL 4");	
	kernel[5]= clCreateKernel(program[5], "cl_merged_blas4", &ret);
		if (ret!=CL_SUCCESS)
		printf("NO KERNEL 5");	
	kernel[6]= clCreateKernel(program[6], "cl_diffusion_north_south_corners", &ret);
		if (ret!=CL_SUCCESS)
		printf("NO KERNEL 6");
	
	ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void*)&x_new_device);
	ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void*)&b_device);
	ret=clSetKernelArg(kernel[0],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[0],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[0],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[0],5,sizeof(int),&ny);
	cl_double dxs = (1000.0*options.dx*options.dx);
	ret=clSetKernelArg(kernel[0],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[0],7,sizeof(double),&alpha_device);
	
	if (ret!=CL_SUCCESS)
		printf("NO ARGS");
	size_t global_item_size1[2]={nx,ny};
	size_t local_item_size1[2]={16,16};

	ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void*)&x_new_device);
	ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void*)&b_device);
	ret=clSetKernelArg(kernel[1],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[1],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[1],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[1],5,sizeof(int),&ny);
	dxs = (1000.0*options.dx*options.dx);
	ret=clSetKernelArg(kernel[1],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[1],7,sizeof(double),&alpha_device);
	
	if (ret!=CL_SUCCESS)
		printf("NO ARGS");
	size_t global_item_size2[2]={(ny-2),2};
	size_t local_item_size2[2]={256,1};
	
	ret=clSetKernelArg(kernel[6],0,sizeof(cl_mem),(void*)&x_new_device);
	ret=clSetKernelArg(kernel[6],1,sizeof(cl_mem),(void*)&b_device);
	ret=clSetKernelArg(kernel[6],2,sizeof(cl_mem),(void*)&x_old_device);
	ret=clSetKernelArg(kernel[6],3,sizeof(cl_mem),(void*)&bnd_device);
	ret=clSetKernelArg(kernel[6],4,sizeof(int),&nx);
	ret=clSetKernelArg(kernel[6],5,sizeof(int),&ny);
	dxs = (1000.0*options.dx*options.dx);
	ret=clSetKernelArg(kernel[6],6,sizeof(double),&dxs);
	ret=clSetKernelArg(kernel[6],7,sizeof(double),&alpha_device);
	
	if (ret!=CL_SUCCESS)
		printf("NO ARGS");
	size_t global_item_size3[2]={(nx),2};
	size_t local_item_size3[2]={256,1};
	
	// start timer
    double timespent = -omp_get_wtime();
	//TODO::EXECUTE THE KERNEL!!!!
	ret=clEnqueueWriteBuffer(command_queue, x_new_device, CL_TRUE, 0 ,(nx*ny)*sizeof(double), x_new,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, x_old_device, CL_TRUE, 0 ,(nx*ny)*sizeof(double), x_old,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, 0 , nx*sizeof(double), bndN,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, nx*sizeof(double) ,(nx)*sizeof(double), bndS,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, (2*nx)*sizeof(double) ,(ny)*sizeof(double), bndW,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, bnd_device, CL_TRUE, (2*nx+ny)*sizeof(double) ,(ny)*sizeof(double), bndE,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, b_device, CL_TRUE, 0 ,(nx*ny)*sizeof(double), b,0, NULL, NULL);
	ret=clEnqueueWriteBuffer(command_queue, deltax_device, CL_TRUE, 0 ,(nx*ny)*sizeof(double), deltax,0, NULL, NULL);
		
    
	for (timestep = 1; timestep <= nt; timestep++)
    {
        // set x_new and x_old to be the solution
        //ss_copy(x_old, x_new, N);
			
		ret=clEnqueueCopyBuffer(command_queue, x_new_device,x_old_device,0,0,N*sizeof(double), 0, NULL, NULL);
		double residual;
        int    converged = 0;
        int    it = 1;
        for ( ; it <= 50; it++)
        {
            // compute residual : requires both x_new and x_old
            //diffusion(x_new, b);
			ret=clSetKernelArg(kernel[0],0,sizeof(cl_mem),(void*)&x_new_device);
			ret=clSetKernelArg(kernel[0],1,sizeof(cl_mem),(void*)&b_device);
			ret=clSetKernelArg(kernel[1],0,sizeof(cl_mem),(void*)&x_new_device);
			ret=clSetKernelArg(kernel[1],1,sizeof(cl_mem),(void*)&b_device);
			ret=clSetKernelArg(kernel[6],0,sizeof(cl_mem),(void*)&x_new_device);
			ret=clSetKernelArg(kernel[6],1,sizeof(cl_mem),(void*)&b_device);
			ret= clEnqueueNDRangeKernel(command_queue, kernel[0], 2, NULL, global_item_size1, local_item_size1, 0 ,NULL, NULL);
			ret= clEnqueueNDRangeKernel(command_queue, kernel[1], 2, NULL, global_item_size2, local_item_size2, 0 ,NULL, NULL);
			ret= clEnqueueNDRangeKernel(command_queue, kernel[6], 2, NULL, global_item_size3, local_item_size3, 0 ,NULL, NULL);
									
			//ss_norm2(b, N);
			ret=clAmdBlasDnrm2(N,residual_device, 0, b_device,0,1, scratchBuff_nrm2,1,&command_queue,0,NULL,NULL);
			ret=clEnqueueReadBuffer(command_queue, residual_device, CL_TRUE, 0 ,sizeof(double), &residual,0, NULL, NULL);
			
            // check for convergence
            if (residual < tolerance)
            {
                converged = 1;
                break;
            }

            // solve linear system to get -deltax
            int cg_converged = 0;
            ss_cg(200, tolerance, &cg_converged,bnd_device,x_new_device,x_old_device,b_device,deltax_device,&command_queue,&context,program,kernel,dxs,alpha_device,nx,ny);
			
            // check that the CG solver converged
            if (!cg_converged) break;

            // update solution
            //ss_axpy(x_new, -1.0, deltax, N);
			cl_double daxpy_alpha = -1.0;
			err=clAmdBlasDaxpy(nx*ny, daxpy_alpha, deltax_device, 0, 1, x_new_device, 0, 1, 1, &command_queue,0, NULL, NULL);
        }
        iters_newton += it;

        // output some statistics
        //if (converged && verbose_output)
        if (converged && verbose_output)
            printf("step %d required %d iterations for residual %E\n", timestep, it, residual);
        if (!converged)
        {
            fprintf(stderr, "step %d ERROR : nonlinear iterations failed to converge\n", timestep);
            break;
        }
    }
	ret=clEnqueueReadBuffer(command_queue, x_new_device, CL_TRUE, 0 ,N*sizeof(double), x_new,0, NULL, NULL);
			
	 // get times
    timespent += omp_get_wtime();
    unsigned long long flops_total = flops_diff + flops_blas1;

	clAmdBlasTeardown();
	ret=clFlush(command_queue);
	ret=clFinish(command_queue);
	
	int loop_temp;
	for (loop_temp=0;loop_temp<7;loop_temp++){
		ret=clReleaseKernel(kernel[loop_temp]);
		ret=clReleaseProgram(program[loop_temp]);
	}
	ret=clReleaseMemObject(x_new_device);
	ret=clReleaseMemObject(x_old_device);
	ret=clReleaseMemObject(bnd_device);
	ret=clReleaseMemObject(b_device);
	ret=clReleaseMemObject(deltax_device);
	ret=clReleaseCommandQueue(command_queue);
	ret=clReleaseContext(context);
	
	
   
    ////////////////////////////////////////////////////////////////////
    // write final solution to BOV file for visualization
    ////////////////////////////////////////////////////////////////////

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
        //fprintf(output, "BYTE_OFFSET: 4\n");
        fprintf(output, "BRICK_SIZE: 1.0 %f 1.0\n", (ny - 1) * options.dx);
        fclose(output);
    }

    // print table sumarizing results
    printf("--------------------------------------------------------------------------------\n");
    printf("simulation took %f seconds\n", timespent);
    printf("%d conjugate gradient iterations\n", (int)iters_cg);
    printf("%d newton iterations\n", (int)iters_newton);
    printf("--------------------------------------------------------------------------------\n");

    // deallocate global fields
    free (x_new);
    free (x_old);
    free (bndN);
    free (bndS);
    free (bndE);
    free (bndW);

    printf("Goodbye!\n");

    return 0;
}

