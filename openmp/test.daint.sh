dim=128
for threads in 1 2 4 8
do
    echo $threads thread
    OMP_NUM_THREADS=$threads aprun -cc none ./main $dim $dim 20 0.005 | grep "simulation took"
done
