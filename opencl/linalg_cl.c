#include "linalg.h"
#include "operators.h"
#include "stats.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
const char * clss_dot[]={
"__kernel void clss_dot(__global const double* x, __gloabal const double* y, const int N, double res)",
"{",
"    double result = 0;",
"	int i=get_global_id(0);",
"    result += x[i] * y[i];",
"    //res = result;",
"}"
};


// sets entries in a vector to value
// x is a vector on length N
// value is th
const char * clss_fill[]={
"__kernel void clss_fill(__global double* x, const double value)",
"{",
"	int i=get_global_id(0);",
"    x[i] = value;",
"}"
};

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
const char clss_axpy[]="																									\
__kernel void clss_axpy(__global double* y, __global const double alpha, __global const double* x, __global const int N) 	\
{																															\
	//int i=get_global_id(0)*64+get_local_id(0);																			\
   // y[i] += alpha * x[i];																									\
}																															\
";

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar

const char * clss_add_scaled_diff[]={
"__kernel void clss_add_scaled_diff(__global double* y, __global const double* x, const double alpha, __global const double* l, __global const double* r, const int N)",
"{",
"	int i=get_global_id(0)*64+get_local_id(0);",
"   y[i] = x[i] + alpha * (l[i] - r[i]);",
"}"
};

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
const char * clss_scaled_diff[]={
"__kernel void clss_scaled_diff(__global double* y, const double alpha,__global const double* l,__global const double* r, const int N)",
"{",
"	int i=get_global_id(0)*64+get_local_id(0);",
"   y[i] = alpha * (l[i] - r[i]);",
"}"
};

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
const char * clss_scale[]={
"__kernel void clss_scale(__global double* y, const double alpha,__global double* x, const int N)",
"{",
"	int i=get_global_id(0)*64+get_local_id(0);",
"    y[i] = alpha * x[i];",
"}"
};


// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n

const char * clss_lcomb[]={
"__kernel void ss_lcomb(__global double* y, const double alpha,__gloabal double* x, const double beta,__global const double* z, const int N)",
"{",
"	int i=get_global_id(0)*64+get_local_id(0);",
"   y[i] = alpha * x[i] + beta * z[i];",
"}"
};