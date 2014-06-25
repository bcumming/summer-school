//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "data.h"
#include "operators.h"
#include "stats.h"
#include <stdio.h>

namespace operators {

/*
void MyPrint(char * text, __m512d a){
	double *temp;
	posix_memalign((void **)&temp, sizeof(__m512d), 8 * sizeof(double));
	_mm512_store_pd((void *)(temp), a);
	printf("%s: ", text);
	for (int i = 0; i < 8; i++)
		printf("%f ", temp[i]);
	printf("\n");
}
*/

inline __m512d _mm512_rotr_pd(__m512d v){
	__m512i iv1 = _mm512_swizzle_epi64(_mm512_castpd_si512(v), _MM_SWIZ_REG_CDAB); 
	return(_mm512_castsi512_pd(_mm512_mask_blend_epi64(85, iv1, _mm512_permute4f128_epi32(iv1, _MM_PERM_CBAD))));
}

inline __m512d _mm512_rotl_pd(__m512d v){
	__m512i iv1 = _mm512_swizzle_epi64(_mm512_castpd_si512(v), _MM_SWIZ_REG_CDAB); 
	return(_mm512_castsi512_pd(_mm512_mask_blend_epi64(255-85, iv1, _mm512_permute4f128_epi32(iv1, _MM_PERM_ADCB))));
}



void diffusion(const data::Field &U, data::Field &S)
{
    using data::options;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;
    int iend  = nx - 1;
    int jend  = ny - 1;
	
	__m512d veccoeff = _mm512_set1_pd(-(4.0 + alpha));
	__m512d vecalpha = _mm512_set1_pd( alpha);
	__m512d vec1 = _mm512_set1_pd(1.0);
	__m512d vecdxs = _mm512_set1_pd(dxs);
    
	// the interior grid points
	#pragma unroll
	#pragma omp parallel for
    for (int j=1; j < jend; j++) {
		//Calculate the number of vectorized iterations
		int ivecend = (iend + 7) / 8;
		#pragma unroll
		for (int ivec = 0; ivec < ivecend; ivec++){
			// i = (ivec * 8 + 8) : (ivec * 8 + 17)
			int i = 8 * ivec; 
			//==============================================
			//Compose U(i+-1, j) as a vector
			__m512d vecu = _mm512_load_pd((void *)&U(i,j));
			//left
			__m512d veculeft = _mm512_rotr_pd(_mm512_mask_blend_pd(128, vecu, _mm512_load_pd((void *)&U(i-8,j))));
			//right
			__m512d vecuright = _mm512_rotl_pd(_mm512_mask_blend_pd(1, vecu, _mm512_load_pd((void *)&U(i+8,j))));
			//==============================================
			//dxs * U(i,j) * (1.0 - U(i,j)) + U(i,j-1)
			__m512d p1 = _mm512_fmadd_pd(vecdxs, _mm512_mul_pd(_mm512_load_pd((void *)&U(i,j)), _mm512_sub_pd(vec1, _mm512_load_pd((void *)&U(i,j)))), _mm512_load_pd((void *)&U(i,j-1)));
			//-(4. + alpha) * U(i,j) + U(i-1,j)
			__m512d p2 = _mm512_fmadd_pd(veccoeff, _mm512_load_pd((void *)&U(i,j)), veculeft);
			//alpha * x_old(i,j) + U(i,j+1)
			__m512d p3 = _mm512_fmadd_pd(vecalpha, _mm512_load_pd((void *)&x_old(i,j)), _mm512_load_pd((void *)&U(i,j+1)));
			//p1 + U(i+1,j)
			__m512d p4 = _mm512_add_pd(p1, vecuright);
			//p2 + p3 + p4
			__m512d res = _mm512_add_pd(_mm512_add_pd(p2, p3), p4);
			//Store to S(i, j)
			if (ivec == 0)
				_mm512_mask_store_pd((void *)(&S(i,j)), 255 - 1, res);
			if ((ivec != 0) && (ivec != ivecend))
				_mm512_store_pd((void *)(&S(i,j)), res);
			if (ivec == ivecend)
				_mm512_mask_store_pd((void *)(&S(i,j)),(1 << (iend - 8 * ivec)) - 1, res);
		}
    }

    // the east boundary
    {
        int i = nx - 1;
		#pragma unroll
		#pragma omp parallel for
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1) + U(i,j+1)
                        + alpha*x_old(i,j) + bndE[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the west boundary
    {
        int i = 0;
		#pragma unroll
		#pragma omp parallel for
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndW[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // north boundary
		#pragma unroll
		#pragma omp parallel for
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j-1)
                        + alpha*x_old(i,j) + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx-1; // NE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndE[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // south boundary
		#pragma unroll
		#pragma omp parallel for
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx - 1; // SE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndE[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // Accumulate the flop counts
    // 8 ops total per point
    stats::flops_diff +=
        + 12 * (nx - 2) * (ny - 2) // interior points
        + 11 * (nx - 2  +  ny - 2) // NESW boundary points
        + 11 * 4;                                  // corner points
}

} // namespace operators
