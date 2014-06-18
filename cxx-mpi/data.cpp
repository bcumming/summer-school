#include "data.h"

#include <stdio.h>

namespace data{
    // fields that hold the solution
    double *x_new = NULL;
    double *x_old = NULL;

    // fields that hold the boundary points
    double *bndN = NULL;
    double *bndE = NULL;
    double *bndS = NULL;
    double *bndW = NULL;

    Discretization options;
}
