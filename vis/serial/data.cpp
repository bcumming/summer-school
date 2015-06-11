#include "data.h"

#include <stdio.h>

namespace data{
// fields that hold the solution
Field x_new;
Field x_old;

// fields that hold the boundary points
Field bndN;
Field bndE;
Field bndS;
Field bndW;

Discretization options;

Field b;
Field deltax;

#ifdef _VISIT_
Field coords_x, coords_y;
bool runMode;
#endif
}
