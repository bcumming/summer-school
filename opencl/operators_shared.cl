#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// modified by Aleksei Ivakhnenko, APC LLC
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#define U(j,i)    shared_U[j][i]
#define S(j,i)    b[(i) + (j)*nx]
#define X(j,i)    x_old[(i) + (j)*nx]


#define bndN(j)		bnd[j]
#define bndS(j)		bnd[nx+j]
#define bndW(j)		bnd[2*nx+j]
#define bndE(j)		bnd[2*nx+ny+j]

__kernel void cl_diffusion_center(__global double* x_new, __global double * b, __global double * x_old, __global double * bnd, __private int nx, __private int ny, __private double dxs, __private double alpha)
{
    //struct discretization_t* options = options;

    //double (*u)[options.nx] = (double(*)[options.nx])up;
    //double (*s)[options.nx] = (double(*)[options.nx])sp;

    //double (*x_old)[options.nx] = (double(*)[options.nx])x_old;
    //double *bndE = bndE, *bndW = bndW;
    //double *bndN = bndN, *bndS = bndS;

    
    //int    iend  = options.nx - 1;
    //int    jend  = options.ny - 1;

    
	int i=get_global_id(0)+1;
	int j=get_global_id(1)+1;
	int local_i=get_local_id(0)+1;
	int local_j=get_local_id(1)+1;
	
	local double shared_U [18][18];
	
	shared_U[local_i][local_j]=x_new[(i) + (j)*nx];
	if (local_i==1){
		shared_U[local_i-1][local_j]=x_new[(i-1) + (j)*nx];
	if (local_j==1)
		shared_U[local_i-1][local_j-1]=x_new[(i-1) + (j-1)*nx];
	if (local_j==16)
		shared_U[local_i-1][local_j+1]=x_new[(i-1) + (j+1)*nx];
	}
	if (local_j==1){
		shared_U[local_i][local_j-1]=x_new[(i) + (j-1)*nx];
	if (local_i==1)
		shared_U[local_i-1][local_j-1]=x_new[(i-1) + (j-1)*nx];
	if (local_i==16)
		shared_U[local_i+1][local_j-1]=x_new[(i+1) + (j-1)*nx];
	}
	if (local_i==16){
		shared_U[local_i+1][local_j]=x_new[(i+1) + (j)*nx];
	if (local_j==1)
		shared_U[local_i+1][local_j-1]=x_new[(i+1) + (j-1)*nx];
	if (local_j==16)
		shared_U[local_i+1][local_j+1]=x_new[(i+1) + (j+1)*nx];
	}
	if (local_j==16){
		shared_U[local_i][local_j+1]=x_new[(i) + (j+1)*nx];
	if (local_i==1)
		shared_U[local_i-1][local_j+1]=x_new[(i-1) + (j+1)*nx];
	if (local_i==16)
		shared_U[local_i+1][local_j+1]=x_new[(i+1) + (j+1)*nx];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
    // the interior grid points
    
	
            S(j, i) = -(4. + alpha)*U(j,i)               // central point
                                    + U(j,i-1) + U(j,i+1) // east and west
                                    + U(j-1,i) + U(j+1,i) // north and south

                                    + alpha*X(j,i)
                                    + dxs*U(j,i)*(1.0 - U(j,i));
									
}

