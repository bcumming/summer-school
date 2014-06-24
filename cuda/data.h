#ifndef DATA_H
#define DATA_H

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct discretization_t
{
	int nx;       // x dimension
	int ny;       // y dimension
	int nt;       // number of time steps
	int N;        // total number of grid points
	double dt;    // time step size
	double dx;    // distance between grid points
	double alpha; // dx^2/(D*dt)
};

namespace cpu
{
	extern struct discretization_t options;
}

namespace gpu
{
	// fields that hold the solution
	extern __device__ double *x_old; // 2d
	extern __device__ double *bndN, *bndE, *bndS, *bndW; // 1d

	extern __constant__ struct discretization_t options;
	
	extern __constant__ cudaDeviceProp props;
}

#endif // DATA_H

