//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "data.h"
#include "operators.h"
#include "stats.h"

using namespace operators;

void operators::diffusion(const double* up, double* sp)
{
	data::discretization_t& options = data::options;
	
	double (*u)[options.nx] = (double(*)[options.nx])up;
	double (*s)[options.nx] = (double(*)[options.nx])sp;

	double (*x_old)[options.nx] = (double(*)[options.nx])data::x_old;
	double *bndE = data::bndE, *bndW = data::bndW;
	double *bndN = data::bndN, *bndS = data::bndS;

	double dxs = 1000. * (options.dx * options.dx);
	double alpha = options.alpha;
	int iend  = options.nx - 1;
	int jend  = options.ny - 1;

	// the interior grid points
	for (int j = 1; j < jend; j++)
		for (int i = 1; i < iend; i++)
		{
			s[j][i] = -(4. + alpha) * u[j][i]               // central point
			                        + u[j][i-1] + u[j][i+1] // east and west
			                        + u[j-1][i] + u[j+1][i] // north and south

			                        + alpha * x_old[j][i]
			                        + dxs * u[j][i] * (1.0 - u[j][i]);
                }

	// the east boundary
	{
		int i = options.nx - 1;
		for (int j = 1; j < jend; j++)
		{
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i-1] + u[j-1][i] + u[j+1][i]
						
						+ alpha*x_old[j][i] + bndE[j]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}
	}
	
	// the west boundary
	{
		int i = 0;
		for (int j = 1; j < jend; j++)
		{
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i+1] + u[j-1][i] + u[j+1][i]
						
						+ alpha * x_old[j][i] + bndW[j]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}
	}

	// the north boundary (plus NE and NW corners)
	{
		int j = options.ny - 1;
		
		{
			int i = 0; // NW corner
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i+1] + u[j-1][i]
						
						+ alpha * x_old[j][i] + bndW[j] + bndN[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}

		// north boundary
		for (int i = 1; i < iend; i++)
		{
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i-1] + u[j][i+1] + u[j-1][i]
						
						+ alpha*x_old[j][i] + bndN[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}

		{
			int i = options.nx; // NE corner
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i-1] + u[j-1][i]
						
						+ alpha * x_old[j][i] + bndE[j] + bndN[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}
	}

	// the south boundary
	{
		int j = 0;
		
		{
			int i = 0; // SW corner
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i+1] + u[j+1][i]
						
						+ alpha * x_old[j][i] + bndW[j] + bndS[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}

		// south boundary
		for (int i = 1; i < iend; i++)
		{
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i-1] + u[j][i+1] + u[j+1][i]
						
						+ alpha * x_old[j][i] + bndS[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}
		
		{
			int i = options.nx - 1; // SE corner
			s[j][i] = -(4. + alpha) * u[j][i]
						+ u[j][i-1] + u[j+1][i]
						
						+ alpha * x_old[j][i] + bndE[j] + bndS[i]
						+ dxs * u[j][i] * (1.0 - u[j][i]);
		}
	}

	// Accumulate the flop counts
	// 8 ops total per point
	stats::flops_diff +=
		+ 12 * (options.nx - 2) * (options.ny - 2) // interior points
		+ 11 * (options.nx - 2  +  options.ny - 2) // NESW boundary points
		+ 11 * 4;                                  // corner points
}

