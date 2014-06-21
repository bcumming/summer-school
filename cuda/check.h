#define CUDA_ERR_CHECK(x)                                  \
    do { cudaError_t err = x; if (err != cudaSuccess) {    \
        fprintf(stderr, "CUDA error %d \"%s\" at %s:%d\n", \
        (int)err, cudaGetErrorString(err),                 \
        __FILE__, __LINE__); exit(-1);                     \
    }} while (0);

