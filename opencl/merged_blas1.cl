#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void cl_merged_blas1(__global double* y, __global double * x, __private double alpha, __global double * l, __global double * r, __global double * p)
{
int i=get_global_id(0);
double temp =x[i]+ alpha * (l[i] - r[i]);
y[i] = temp;
p[i] =temp;
}