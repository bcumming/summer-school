#include "data.h"

#include <stdio.h>

// fields that hold the solution
double *data::x_new = NULL, *data::x_old = NULL; // 2d
double *data::bndN = NULL, *data::bndE = NULL, *data::bndS = NULL, *data::bndW = NULL;

data::discretization_t data::options;

