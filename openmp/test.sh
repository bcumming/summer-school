for threads in 1 2 4 8
do
    OMP_NUM_THREADS=$threads aprun -cc none ./main 128 128 20 0.005 | grep "simulation took"
done
