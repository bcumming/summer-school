## Finite-difference solver for non-linear 2D diffusion-reaction problem

### How to build

Each of the components (serial, OpenMP, etc...) is designed to be self-contained. Building a component on a Cray system should be as simple as (assuming that either the GNU or Cray programming environment is loaded):

```
$ cd <component>
$ make
```

Build notes specific to individual components is given below where relevant

#### OpenMP component

The OpenMP Fortran component has been tested with both Cray and GNU compilers on XK7, XC30 systems.

### How to test

```
$ cd cxx
$ ./main 128 128 10000 1
```

Open resulting output.bov in Paraview:

![paraview.png](images/paraview.png)
