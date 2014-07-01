#ifndef DATA_H
#define DATA_H
#include "immintrin.h"
#include <assert.h>
#include <stdlib.h>
#include <sys/mman.h>

namespace data
{

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ALIGN_TO_PAGE_SIZE(x) \
    (((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int nt;       // number of time steps
    int N;        // total number of grid points
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

extern Discretization options;

// thin wrapper around a pointer that can be accessed as either a 2D or 1D array
// Field has dimension xdim * ydim in 2D, or length=xdim*ydim in 1D
class Field {
    public:
    // default constructor
    Field()
    :   ptr_(0), xdim_(0), ydim_(0)
    {};

    // constructor
    Field(int xdim, int ydim)
    :   xdim_(xdim), ydim_(ydim)
    {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        ptr_ = (double *) malloc_huge_pages(xdim * ydim *sizeof(double));
        //posix_memalign((void **)&ptr_, sizeof(__m512d), xdim*ydim*sizeof(double));
    };

    // destructor
    ~Field() {
        FieldFree();
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        FieldFree();
        ptr_ = (double *) malloc_huge_pages(xdim * ydim *sizeof(double));
        //posix_memalign((void **)&ptr_, sizeof(__m512d), xdim*ydim*sizeof(double));
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        xdim_ = xdim;
        ydim_ = ydim;
    }

    double*       data()       { return ptr_; }
    const double* data() const { return ptr_; }

    // access via (i,j) pair
    inline double&       operator() (int i, int j)        {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }
    inline double const& operator() (int i, int j) const  {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_ && j>=0 && j<ydim_);
        #endif
        return ptr_[i+j*xdim_];
    }

    // access as a 1D field
    inline double      & operator[] (int i) {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }
    inline double const& operator[] (int i) const {
        #ifdef DEBUG
        assert(i>=0 && i<xdim_*ydim_);
        #endif
        return ptr_[i];
    }

    int xdim()   const { return xdim_; }
    int ydim()   const { return ydim_; }
    int length() const { return xdim_*ydim_; }

    private:

    void* malloc_huge_pages(size_t size){
        // Use 1 extra page to store allocation metadata
        // (libhugetlbfs is more efficient in this regard)
        size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
        char *ptr = (char *)mmap(NULL, real_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);

        if (ptr == MAP_FAILED) {
            // The mmap() call failed. Try to malloc instead
            ptr = (char *)malloc(real_size);
            if (ptr == NULL) return NULL;
            real_size = 0;
        }

        // Save real_size since mmunmap() requires a size parameter
        *((size_t *)ptr) = real_size;
        // Skip the page with metadata
        return ptr + HUGE_PAGE_SIZE;
    }

    void free_huge_pages(void *ptr){
        if (ptr == NULL) return;

        // Jump back to the page with metadata
        void *real_ptr = (char *)ptr - HUGE_PAGE_SIZE;

        // Read the original allocation size
        size_t real_size = *((size_t *)real_ptr);
        assert(real_size % HUGE_PAGE_SIZE == 0);

        if (real_size != 0)
            // The memory was allocated via mmap()
            // and must be deallocated via munmap()
            munmap(real_ptr, real_size);
        else
            // The memory was allocated via malloc()
            // and must be deallocated via free()
            free(real_ptr);
    }
    
    void FieldFree() {
        //if(ptr_) delete[] ptr_;
        if(ptr_) free_huge_pages((void *) ptr_);
        ptr_ = 0;
    }

    double* ptr_;
    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new, x_old; // 2d
extern Field bndN, bndE, bndS, bndW; // 1d

}

#endif // DATA_H

