#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void cl_merged_blas2(__global double* y, __global double * x, __private double alpha)
{
int i=get_global_id(0);
double temp =x[i];
y[i]=temp*alpha;
}