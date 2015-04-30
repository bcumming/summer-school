#!/bin/bash

for threads in 1 2 4 8
do
    export OMP_NUM_THREADS=$threads
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "                      $threads THREADS"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    aprun -cc none ./main 256 256 50 0.001 | grep second
done
