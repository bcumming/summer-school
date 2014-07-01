#!/bin/bash

export KMP_AFFINITY=granularity=fine,balanced
export KMP_PLACE_THREADS="56c,4t"
export OMP_NUM_THREADS=224
export LD_LIBRARY_PATH=/home/ashevchenko/lib
./main 512 512 400 0.01
