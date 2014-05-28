export MPICH_RDMA_ENABLED_CUDA=0

dim=512
nprocx=1
nprocy=1
nt=50
echo "============================================="
echo "weak scaling $dim*$dim for $nt steps"
echo "============================================="
for i in 1 2 3 4 5
do
    xdim=$[$nprocx * $dim]
    ydim=$[$nprocy * $dim]
    nnodes=$[$nprocx * $nprocy]
    nproc=$[8 * $nprocx * $nprocy]
    steps_per_second=`aprun -n $nproc  -N 8 ./main $xdim $ydim $nt 0.0025 | grep "per second" | awk '{printf("%9.1f", $5)}'`
    echo "CG iterations per second = $steps_per_second :: $nnodes nodes ($xdim*$ydim)"
    if [ "$nprocx" -lt "$nprocy" ]
    then
        nprocx=$nprocy
    else
        nprocy=$[$nprocy * 2]
    fi
done

nt=400
dim=1024
nt=100
echo "============================================="
echo "strong scaling $dim*$dim for $nt steps"
echo "============================================="
nproc=8
for i in 1 2 3 4 5
do
    steps_per_second=`aprun -n $nproc -N 8 ./main $dim $dim $nt 0.0025 | grep "per second" | awk '{printf("%9.1f", $5)}'`
    echo "CG iterations per second = $steps_per_second :: $nproc nodes ($dim*$dim)"
    nproc=$[$nproc * 2];
done

