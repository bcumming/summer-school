#pragma once

#include <cassert>
#include <cuda.h>

namespace data
{

// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int nt;       // number of time steps
    //TODO wherever you see this cause an error, double check if we want global or local value
    //int N;        // total number of grid points
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

struct SubDomain
{
    // initialize a subdomain
    void init(int, int, Discretization&);

    // print subdomain information
    void print();

    // i and j dimensions of the global decomposition
    int ndomx;
    int ndomy;

    // the i and j index of this sub-domain
    int domx;
    int domy;

    // the i and j bounding box of this sub-domain
    int startx;
    int starty;
    int endx;
    int endy;

    // the rank of neighbouring domains
    int neighbour_north;
    int neighbour_east;
    int neighbour_south;
    int neighbour_west;

    // mpi info
    int size;
    int rank;

    // x and y dimension in grid points of the sub-domain
    int nx;
    int ny;

    // total number of grid points
    int N;
};

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
        //ptr_ = new double[xdim*ydim];
	cudaMalloc((void**)&ptr_, sizeof(double)*xdim*ydim);        
	// do first touch
        //fill(0.);
	cudaMemset(ptr_,0,sizeof(double)*xdim,ydim);
    };

    // destructor
    ~Field() {
        free();
    }

    void init(int xdim, int ydim) {
        #ifdef DEBUG
        assert(xdim>0 && ydim>0);
        #endif
        free();
        cudaMalloc((void**)&ptr_,sizeof(double)*(xdim*ydim));
		//ptr_ = new double[xdim*ydim];
        xdim_ = xdim;
        ydim_ = ydim;

        // do first touch
        //fill(0.);
	cudaMemset(ptr_,0,sizeof(double)*xdim,ydim);
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

    // set to a constant value
    void fill(double val) {
        #pragma omp parallel for
        for(int i=0; i<xdim_*ydim_; ++i)
            ptr_[i] = val;
    }

    void free() {
        if(ptr_) delete[] ptr_;
        ptr_ = 0;
    }

    double* ptr_;
    int xdim_;
    int ydim_;
};

// fields that hold the solution
extern Field x_new; // 2d
extern Field x_old; // 2d

// fields that hold the boundary values
extern Field bndN; // 1d
extern Field bndE; // 1d
extern Field bndS; // 1d
extern Field bndW; // 1d

// buffers used in boundary exchange
extern Field buffN;
extern Field buffE;
extern Field buffS;
extern Field buffW;

extern Discretization options;
extern SubDomain      domain;

namespace cpu
{
	extern struct Discretization options;

	extern cudaDeviceProp props;
}
/*#include <thrust/extrema.h>

namespace gpu
{
	// We redefine dim3 under namespace, because the default one has
	// constructors, which is not allowed for types device variables
	// (dim3 is used as device vars type below to keep kernel compute
	// grid configuration).
	struct dim3
	{
		unsigned int x, y, z;
		
		__device__ operator ::dim3()
		{
			return ::dim3(x, y, z);
		}
	};

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

	static __device__ __inline__ double2 __ld(const double2 *ptr)
	{
		double2 ret;
		asm volatile ("ld.global.v2.f64 {%0,%1}, [%2];"  : "=d"(ret.x), "=d"(ret.y) : __LDG_PTR (ptr));
		return ret;
	}
#undef __LDG_PTR

	static __device__ __inline__ void __stcs(const double2 *ptr, const double2& val)
	{
		asm volatile ("st.global.cs.v2.f64 [%0], {%1,%2};"  : : "l"(ptr), "d"(val.x), "d"(val.y));
	}
	
	union double1
	{
		double e;
		double v[1];

		inline static __device__ double1 ld(const double* src, const int i)
		{
			double1 dst;
			dst.e = src[i];
			return dst;
		}

		inline static __device__ double1 ldg(const double* src, const int i)
		{
			double1 dst;
			dst.e = __ldg(&src[i]);
			return dst;
		}
		
		inline static __device__ void stcs(double* dst, const int i, const double1& src)
		{
			dst[i] = src.e;
		}
	};
	
	union double2
	{
		::double2 e;
		double v[2];

		inline static __device__ double2 ld(const double* src, const int i)
		{
			double2 dst;
			dst.e = __ld(&((const double2*)src)[i].e);
			return dst;
		}

		inline static __device__ double2 ldg(const double* src, const int i)
		{
			double2 dst;
			dst.e = __ldg(&((const double2*)src)[i].e);
			return dst;
		}
		
		inline static __device__ void stcs(double* dst, const int i, const double2& src)
		{
			__stcs(&((const double2*)dst)[i].e, src.e);
		}
	};
	
	union double4
	{
		::double4 e;
		::double2 e2[2];
		double v[4];

		inline static __device__ double4 ld(const double* src, const int i)
		{
			double4 dst;
			dst.e2[0] = __ld(&((const double4*)src)[i].e2[0]);
			dst.e2[1] = __ld(&((const double4*)src)[i].e2[1]);
			return dst;
		}

		inline static __device__ double4 ldg(const double* src, const int i)
		{
			double4 dst;
			dst.e2[0] = __ldg(&((const double4*)src)[i].e2[0]);
			dst.e2[1] = __ldg(&((const double4*)src)[i].e2[1]);
			return dst;
		}
		
		inline static __device__ void stcs(double* dst, const int i, const double4& src)
		{
			__stcs(&((const double4*)dst)[i].e2[0], src.e2[0]);
			__stcs(&((const double4*)dst)[i].e2[1], src.e2[1]);
		}
	};
	
	// Use Thrust occupancy calculator to determine the best size of block.
	template<typename T>
	inline size_t get_optimal_szblock(T kernel)
	{
		using namespace gpu;
		using namespace thrust::system::cuda::detail;

		struct function_attributes_t attrs;
		{
			cudaFuncAttributes funcAttrs;
			CUDA_ERR_CHECK(cudaFuncGetAttributes(&funcAttrs, kernel));
			attrs.constSizeBytes = funcAttrs.constSizeBytes;
			attrs.localSizeBytes = funcAttrs.localSizeBytes;
			attrs.maxThreadsPerBlock = funcAttrs.maxThreadsPerBlock;
			attrs.numRegs = funcAttrs.numRegs;
			attrs.sharedSizeBytes = funcAttrs.sharedSizeBytes;
		}
		struct device_properties_t props;
		{
			props.major = cpu::props.major;
			memcpy(&props.maxGridSize, &cpu::props.maxGridSize, sizeof(int) * 3);
			props.maxThreadsPerBlock = cpu::props.maxThreadsPerBlock;
			props.maxThreadsPerMultiProcessor = cpu::props.maxThreadsPerMultiProcessor;
			props.minor = cpu::props.minor;
			props.multiProcessorCount = cpu::props.multiProcessorCount;
			props.regsPerBlock = cpu::props.regsPerBlock;
			props.sharedMemPerBlock = cpu::props.sharedMemPerBlock;
			props.warpSize = cpu::props.warpSize;
		}
		return block_size_with_maximum_potential_occupancy(attrs, props);
	}

	struct block_size_to_dynamic_smem_size : public thrust::unary_function<size_t, size_t>
	{
		float operator()(size_t szblock) { return szblock * sizeof(double); }
	};

	// Use Thrust occupancy calculator to determine the best size of block
	// for a kernel which uses dynamic shared memory
	template<typename T, typename block_size_to_dynamic_smem_size>
	inline size_t get_optimal_szblock(T kernel)
	{
		using namespace gpu;
		using namespace thrust::system::cuda::detail;

		struct function_attributes_t attrs;
		{
			cudaFuncAttributes funcAttrs;
			CUDA_ERR_CHECK(cudaFuncGetAttributes(&funcAttrs, kernel));
			attrs.constSizeBytes = funcAttrs.constSizeBytes;
			attrs.localSizeBytes = funcAttrs.localSizeBytes;
			attrs.maxThreadsPerBlock = funcAttrs.maxThreadsPerBlock;
			attrs.numRegs = funcAttrs.numRegs;
			attrs.sharedSizeBytes = funcAttrs.sharedSizeBytes;
		}
		struct device_properties_t props;
		{
			props.major = cpu::props.major;
			memcpy(&props.maxGridSize, &cpu::props.maxGridSize, sizeof(int) * 3);
			props.maxThreadsPerBlock = cpu::props.maxThreadsPerBlock;
			props.maxThreadsPerMultiProcessor = cpu::props.maxThreadsPerMultiProcessor;
			props.minor = cpu::props.minor;
			props.multiProcessorCount = cpu::props.multiProcessorCount;
			props.regsPerBlock = cpu::props.regsPerBlock;
			props.sharedMemPerBlock = cpu::props.sharedMemPerBlock;
			props.warpSize = cpu::props.warpSize;
		}
		return block_size_with_maximum_potential_occupancy(attrs, props, block_size_to_dynamic_smem_size());
	}

	template<typename T>
	inline void get_optimal_grid_block_config(T kernel,
		int nx, int ny, size_t szblock, dim3* grid, dim3* blocks)
	{
		grid->x = 1; grid->y = 1; grid->z = 1;
		blocks->x = 1; blocks->y = 1; blocks->z = 1;

		if (szblock > nx)
		{
			blocks->x = nx;
			blocks->y = min(ny, (int)szblock / blocks->x);
			grid->y = ny / blocks->y;
			if (ny % blocks->y) grid->y++;
		}
		else
		{
			blocks->x = szblock;
			grid->x = nx / blocks->x;
			if (nx % blocks->x) grid->x++;
			grid->y = ny;
		}
	}

	#define determine_optimal_grid_block_config(kernel_name, vector, nx, ny) \
	{ \
		{ \
			using namespace gpu::kernel_name##_kernel; \
			gpu::config_t c; \
			size_t szblock = gpu::get_optimal_szblock(kernel<vector, gpu::double##vector>); \
			gpu::get_optimal_grid_block_config(kernel<vector, gpu::double##vector>, ((nx) / (vector)), ny, szblock, &c.grid, &c.block); \
			CUDA_ERR_CHECK(cudaMemcpyToSymbol(config_c, &c, sizeof(gpu::config_t))); \
		} \
	}

	#define determine_optimal_grid_block_config_reduction(kernel_name, vector, nx, c, i) \
	{ \
		{ \
			using namespace gpu::kernel_name##_kernel; \
			size_t szblock = gpu::get_optimal_szblock(kernel<vector, gpu::double##vector>); \
			gpu::get_optimal_grid_block_config(kernel<vector, gpu::double##vector>, ((nx) / (vector)), 1, szblock, &c.grid, &c.block); \
			CUDA_ERR_CHECK(cudaMemcpyToSymbol(configs_c, &c, sizeof(gpu::config_t), i * sizeof(gpu::config_t))); \
		} \
	}

	#define MAX_CONFIGS 4

	#define determine_optimal_grid_block_configs_reduction(kernel_name, vector, n) \
	{ \
		int length = n; \
		using namespace gpu; \
		int iconfig = 0; \
		config_t config; \
		determine_optimal_grid_block_config_reduction(kernel_name, vector, length / 2 + length % 2, config, iconfig); \
		iconfig++; \
		for (int szbuffer = config.grid.x ; szbuffer != 1; szbuffer = config.grid.x) \
		{ \
			length = szbuffer / 2 + szbuffer % 2; \
			determine_optimal_grid_block_config_reduction(kernel_name, vector, length / 2 + length % 2, config, iconfig); \
			iconfig++; \
		} \
	}

	template<typename T>
	inline T get_value(T& var)
	{
		T* ptr;
		CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&ptr, var));
		T value;
		CUDA_ERR_CHECK(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost));
		return value;
	}

	typedef struct __attribute__((packed)) { dim3 grid, block; } config_t;

	// round up to the power of 2
	template<typename T>
	inline __device__ T roundPow2(T ptr, int pow2)
	{
		size_t number = (size_t)ptr;
		pow2--;
		pow2 = 0x01 << pow2;
		pow2--;
		number--;
		number = number | pow2;
		number++;
		return (T)number;
	}
}
*/


} // namespace data

