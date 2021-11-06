/*
 * Copyright (C) 2021 Matthias Kirchhart
 *
 * This file is part of solidfmm, a C++ library of operations on the solid
 * harmonics for use in fast multipole methods.
 *
 * solidfmm is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * solidfmm is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * solidfmm; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#include <solidfmm/microkernel_avx2.hpp>

#ifdef __x86_64__
#pragma GCC target   ("avx,avx2")
#pragma GCC optimize ("O2")
#pragma clang attribute push(__attribute__((target("avx,avx2"))), apply_to=any(function))
#include <immintrin.h>

#include <stdexcept>

namespace solidfmm
{

bool microkernel_float_avx2::available() noexcept
{
    int result_avx   = microkernel_float_avx::available();
    int result_avx2  = __builtin_cpu_supports( "avx2" );

    return result_avx && result_avx2;
}

microkernel_float_avx2::microkernel_float_avx2()
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_float_avx2: "
                                   "AVX2 instructions unavailable on this CPU." };
}

bool microkernel_double_avx2::available() noexcept
{
    int result_avx   = microkernel_double_avx::available();
    int result_avx2  = __builtin_cpu_supports( "avx2" );

    return result_avx && result_avx2;
}

microkernel_double_avx2::microkernel_double_avx2()
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_double_avx2: "
                                   "AVX2 instructions unavailable on this CPU." };
}


void microkernel_float_avx2::
buf2solid( const float   *real_in, const float *imag_in, 
                 float  **solids,  const size_t *P, size_t n ) const noexcept
{
    // We can use AVX2 gather operations to try speeding things up a little.
    // On my machine, however, there was no significant difference. Maybe
    // Other processors fare better. Unfortunately, there are no scatter operations,
    // so we cannot do much for the other way around, solid2buf.

    // Strides for for the input buffers.
    __m256i idx = _mm256_set_epi32( 112, 96, 80, 64, 48, 32, 16, 0 );
    for ( size_t l = 0; l < 16; ++l )
    {
        if ( n < P[l] )
        {
            size_t m = 0;
            for ( ; m + 8 <= n; m += 8 )
            {
                __m256 solid_re = _mm256_loadu_ps( solids[l] + (n  )*(n+1) + m );
                __m256 solid_im = _mm256_loadu_ps( solids[l] + (n+1)*(n+1) + m );

                __m256 buf_re = _mm256_i32gather_ps( real_in + m*16 + l, idx, sizeof(float) );
                __m256 buf_im = _mm256_i32gather_ps( imag_in + m*16 + l, idx, sizeof(float) );

                solid_re = _mm256_add_ps( solid_re, buf_re );
                solid_im = _mm256_add_ps( solid_im, buf_im );
                
                _mm256_storeu_ps( solids[l] + (n  )*(n+1) + m, solid_re );
                _mm256_storeu_ps( solids[l] + (n+1)*(n+1) + m, solid_im );
            } 

            for ( ; m <= n; ++m )
            {
                solids[l][ (n  )*(n+1) + m ] += real_in[ m*16 + l ];
                solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*16 + l ];
            } 
        }
    }
}

void microkernel_double_avx2::
buf2solid( const double   *real_in, const double *imag_in, 
                 double  **solids,  const size_t *P, size_t n ) const noexcept
{
    // We can use AVX2 gather operations to try speeding things up a little.
    // On my machine, however, there was no significant difference. Maybe
    // Other processors fare better. Unfortunately, there are no scatter operations,
    // so we cannot do much for the other way around, solid2buf.

    // Strides for for the input buffers.
    __m256i idx = _mm256_set_epi64x( 24, 16, 8, 0 );
    for ( size_t l = 0; l < 8; ++l )
    {
        if ( n < P[l] )
        {
            size_t m = 0;
            for ( ; m + 4 <= n; m += 4 )
            {
                __m256d solid_re = _mm256_loadu_pd( solids[l] + (n  )*(n+1) + m );
                __m256d solid_im = _mm256_loadu_pd( solids[l] + (n+1)*(n+1) + m );

                __m256d buf_re = _mm256_i64gather_pd( real_in + m*8 + l, idx, sizeof(double) );
                __m256d buf_im = _mm256_i64gather_pd( imag_in + m*8 + l, idx, sizeof(double) );

                solid_re = _mm256_add_pd( solid_re, buf_re );
                solid_im = _mm256_add_pd( solid_im, buf_im );
                
                _mm256_storeu_pd( solids[l] + (n  )*(n+1) + m, solid_re );
                _mm256_storeu_pd( solids[l] + (n+1)*(n+1) + m, solid_im );
            } 

            for ( ; m <= n; ++m )
            {
                solids[l][ (n  )*(n+1) + m ] += real_in[ m*8 + l ];
                solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*8 + l ];
            } 
        }
    }
}

}

#pragma clang attribute pop
#endif

