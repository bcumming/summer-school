#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void cl_merged_blas3(__global double* y, __global double * x, __private double alpha, __private double beta, __global double *z)
{
int i=get_global_id(0);
y[i]=alpha* x[i]+ beta*z[i];
}
