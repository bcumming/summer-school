#include "data.h" 
using namespace data;

/* __device__ inline double min (double x, double y){
if (x>y)
	return x;
else
	return y;
 } 
 */
 __global__ void init_x_new(double * x_new,int nx,int ny, int dx, int ny_domain, int starty, int startx)
 {
	double temp=0.0;
    double xc = 1.0 / 4.0;
    double yc = (ny_domain - 1) * dx / 4;
    double radius = min(xc, yc) / 2.0;
    int indexX=threadIdx.x+blockIdx.x*blockDim.x;
    int indexY=threadIdx.y+blockIdx.y*blockDim.y;
	if ((indexX>=nx)||(indexY>=ny))
		return;
	
	double y = (starty-2+indexY)*dx;
    double x = (startx-2+indexX)*dx;
	if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
		temp= 0.1;
    		x_new[indexX+ indexY*nx] =temp;
	//for (int j = domain.starty-1; j < domain.endy; j++)
    //{
    //    double y = (j - 1) * options.dx;
    //    for (int i = domain.startx-1; i < domain.endx; i++)
    //    {
    //        double x = (i - 1) * options.dx;
    //        if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius)
    //            x_new(i-domain.startx+1, j-domain.starty+1) = 0.1;
    //    }
    //}
 }
