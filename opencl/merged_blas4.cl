#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void cl_merged_blas4(__global double* y, __private double alpha, __global double * l, __global double * r)
{
int i=get_global_id(0);
y[i] =alpha * (l[i] - r[i]);
}