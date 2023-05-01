/*
 * Copyright (C) 2021, 2022 Matthias Kirchhart
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
#include <solidfmm/microkernel_avx512f.hpp>

#ifdef __x86_64__

#ifdef __llvm__
#pragma clang attribute push(__attribute__((target("avx512f"))), apply_to=any(function))
#elif __GNUG__
#pragma GCC target   ("avx512f")
#pragma GCC optimize ("O2")
#endif


#include <immintrin.h>
#include <stdexcept>

namespace solidfmm
{

///////////////////////
// Single Precision. //
///////////////////////

bool microkernel_float_avx512f::available() noexcept
{
    return __builtin_cpu_supports( "avx512f" );
}

microkernel_float_avx512f::microkernel_float_avx512f(): microkernel { 14, 32, 64 }
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_float_avx512f: "
                                   "AVX-512F instructions unavailable on this CPU." };
}

void microkernel_float_avx512f::
euler( const float *x,   const float *y, const float *z,
       float *r,         float *rinv,
       float *cos_alpha, float *sin_alpha,
       float *cos_beta,  float *sin_beta,
       size_t k ) const noexcept
{
    __m512 ones  = _mm512_set1_ps(1.0f);
    __m512 zeros = _mm512_setzero_ps();
    __m512 r0, r1, rinv0, rinv1; 
    __m512 ca0, sa0, ca1, sa1;   // cos/sin alpha
    __m512 cb0, sb0, cb1, sb1;   // cos/sin beta

    if ( k-- )
    {
        _mm512_store_ps( r             , ones  );
        _mm512_store_ps( r         + 16, ones  );
        _mm512_store_ps( rinv          , ones  );
        _mm512_store_ps( rinv      + 16, ones  );
        _mm512_store_ps( cos_alpha     , ones  );
        _mm512_store_ps( cos_alpha + 16, ones  );
        _mm512_store_ps( sin_alpha     , zeros );
        _mm512_store_ps( sin_alpha + 16, zeros );
        _mm512_store_ps( cos_beta      , ones  );
        _mm512_store_ps( cos_beta  + 16, ones  );
        _mm512_store_ps( sin_beta      , zeros );
        _mm512_store_ps( sin_beta  + 16, zeros );

        r         = r         + 32;
        rinv      = rinv      + 32;
        cos_alpha = cos_alpha + 32;
        sin_alpha = sin_alpha + 32;
        cos_beta  = cos_beta  + 32;
        sin_beta  = sin_beta  + 32;
    }
    else return;

    if ( k-- )
    {
        __m512 x0, x1, y0, y1, z0, z1;
        __m512 rxyinv0, rxyinv1, rxy0, rxy1;

        x0   = _mm512_load_ps(x+ 0); 
        x1   = _mm512_load_ps(x+16); 
        y0   = _mm512_load_ps(y+ 0); 
        y1   = _mm512_load_ps(y+16); 
        z0   = _mm512_load_ps(z+ 0); 
        z1   = _mm512_load_ps(z+16); 

        rxy0 = _mm512_mul_ps(x0,x0);
        rxy1 = _mm512_mul_ps(x1,x1);
        rxy0 = _mm512_fmadd_ps(y0,y0,rxy0);
        rxy1 = _mm512_fmadd_ps(y1,y1,rxy1);
        r0   = _mm512_fmadd_ps(z0,z0,rxy0);
        r1   = _mm512_fmadd_ps(z1,z1,rxy1);

        rxy0 = _mm512_sqrt_ps( rxy0 );
        rxy1 = _mm512_sqrt_ps( rxy1 );
        r0   = _mm512_sqrt_ps( r0 );
        r1   = _mm512_sqrt_ps( r1 );

        rxyinv0 = _mm512_div_ps( ones, rxy0 );
        rxyinv1 = _mm512_div_ps( ones, rxy1 );
        rinv0   = _mm512_div_ps( ones, r0 );
        rinv1   = _mm512_div_ps( ones, r1 );

        ca0 = _mm512_mul_ps( y0, rxyinv0 );
        ca1 = _mm512_mul_ps( y1, rxyinv1 );
        sa0 = _mm512_mul_ps( x0, rxyinv0 );
        sa1 = _mm512_mul_ps( x1, rxyinv1 );

        __mmask16 iszero0 = _mm512_cmp_ps_mask( rxy0, zeros, _CMP_EQ_OQ );
        __mmask16 iszero1 = _mm512_cmp_ps_mask( rxy1, zeros, _CMP_EQ_OQ );
        ca0 = _mm512_mask_mov_ps( ca0, iszero0, ones );
        ca1 = _mm512_mask_mov_ps( ca1, iszero1, ones );
        sa0 = _mm512_mask_mov_ps( sa0, iszero0, zeros );
        sa1 = _mm512_mask_mov_ps( sa1, iszero1, zeros );

        cb0 = _mm512_mul_ps( z0, rinv0 );
        cb1 = _mm512_mul_ps( z1, rinv1 );
        sb0 = _mm512_fnmadd_ps( rxy0, rinv0, zeros );
        sb1 = _mm512_fnmadd_ps( rxy1, rinv1, zeros );

        _mm512_store_ps( r             , r0 );
        _mm512_store_ps( r         + 16, r1 );
        _mm512_store_ps( rinv          , rinv0 );
        _mm512_store_ps( rinv      + 16, rinv1 );
        _mm512_store_ps( cos_alpha     , ca0 );
        _mm512_store_ps( cos_alpha + 16, ca1 );
        _mm512_store_ps( sin_alpha     , sa0 );
        _mm512_store_ps( sin_alpha + 16, sa1 );
        _mm512_store_ps( cos_beta      , cb0 );
        _mm512_store_ps( cos_beta  + 16, cb1 );
        _mm512_store_ps( sin_beta      , sb0 );
        _mm512_store_ps( sin_beta  + 16, sb1 );
    }
    else return;

    while ( k-- )
    {
        __m512 rn0, rn1, rninv0, rninv1;
        __m512 cn0, sn0, cn1, sn1, res0, res1;
       
        // r^{n+1}    = r^{ n} * r; 
        // r^{-(n+1)} = r^{-n} * r^{-1}; 
        rn0    = _mm512_load_ps( r     ); 
        rn1    = _mm512_load_ps( r + 16 ); 
        rninv0 = _mm512_load_ps( rinv     ); 
        rninv1 = _mm512_load_ps( rinv + 16 ); 
        
        rn0    = _mm512_mul_ps( rn0, r0 );
        rn1    = _mm512_mul_ps( rn1, r1 );
        rninv0 = _mm512_mul_ps( rninv0, rinv0 );
        rninv1 = _mm512_mul_ps( rninv1, rinv1 );
        _mm512_store_ps( r    + 32, rn0 );
        _mm512_store_ps( r    + 48, rn1 );
        _mm512_store_ps( rinv + 32, rninv0 );
        _mm512_store_ps( rinv + 48, rninv1 );

        // cos(alpha*(n+1)) = cos(alpha*n)*cos(alpha) - sin(alpha*n)*sin(alpha);
        // sin(alpha*(n+1)) = sin(alpha*n)*cos(alpha) + cos(alpha*n)*sin(alpha);
        cn0  = _mm512_load_ps( cos_alpha      );
        cn1  = _mm512_load_ps( cos_alpha + 16 );
        sn0  = _mm512_load_ps( sin_alpha      );
        sn1  = _mm512_load_ps( sin_alpha + 16 );
        res0 = _mm512_mul_ps(cn0,ca0);
        res1 = _mm512_mul_ps(cn1,ca1);
        res0 = _mm512_fnmadd_ps(sn0,sa0,res0);
        res1 = _mm512_fnmadd_ps(sn1,sa1,res1);
        _mm512_store_ps( cos_alpha + 32, res0 );
        _mm512_store_ps( cos_alpha + 48, res1 );
        res0 = _mm512_mul_ps(sn0,ca0);
        res1 = _mm512_mul_ps(sn1,ca1);
        res0 = _mm512_fmadd_ps(cn0,sa0,res0);
        res1 = _mm512_fmadd_ps(cn1,sa1,res1);
        _mm512_store_ps( sin_alpha + 32, res0 );
        _mm512_store_ps( sin_alpha + 48, res1 );

        // cos(beta*(n+1)) = cos(beta*n)*cos(beta) - sin(beta*n)*sin(beta);
        // sin(beta*(n+1)) = sin(beta*n)*cos(beta) + cos(beta*n)*sin(beta);
        cn0  = _mm512_load_ps( cos_beta      );
        cn1  = _mm512_load_ps( cos_beta + 16 );
        sn0  = _mm512_load_ps( sin_beta      );
        sn1  = _mm512_load_ps( sin_beta + 16 );
        res0 = _mm512_mul_ps(cn0,cb0);
        res1 = _mm512_mul_ps(cn1,cb1);
        res0 = _mm512_fnmadd_ps(sn0,sb0,res0);
        res1 = _mm512_fnmadd_ps(sn1,sb1,res1);
        _mm512_store_ps( cos_beta + 32, res0 );
        _mm512_store_ps( cos_beta + 48, res1 );
        res0 = _mm512_mul_ps(sn0,cb0);
        res1 = _mm512_mul_ps(sn1,cb1);
        res0 = _mm512_fmadd_ps(cn0,sb0,res0);
        res1 = _mm512_fmadd_ps(cn1,sb1,res1);
        _mm512_store_ps( sin_beta + 32, res0 );
        _mm512_store_ps( sin_beta + 48, res1 );
        
        r         = r         + 32;
        rinv      = rinv      + 32;
        cos_alpha = cos_alpha + 32;
        sin_alpha = sin_alpha + 32;
        cos_beta  = cos_beta  + 32;
        sin_beta  = sin_beta  + 32;
    }
}

void microkernel_float_avx512f::
rotscale( const float *cos,      const float *sin,      const float *scale,
          const float *real_in,  const float *imag_in,
                float *__restrict__ real_out,      
                float *__restrict__ imag_out,
          size_t k, bool forward ) const noexcept
{
    __m512 scale0, scale1;
    __m512 cos0, cos1, sin0, sin1;
    __m512 rin0, rin1;
    __m512 iin0, iin1;
    __m512 tmp0, tmp1;

    scale0 = _mm512_load_ps( scale      );
    scale1 = _mm512_load_ps( scale + 16 );

    if ( forward )
    {
        while ( k-- )
        {
            cos0 = _mm512_load_ps( cos      );
            cos1 = _mm512_load_ps( cos + 16 );
            sin0 = _mm512_load_ps( sin      );
            sin1 = _mm512_load_ps( sin + 16 );
            rin0 = _mm512_load_ps( real_in      );
            rin1 = _mm512_load_ps( real_in + 16 );
            iin0 = _mm512_load_ps( imag_in      );
            iin1 = _mm512_load_ps( imag_in + 16 );

            tmp0 = _mm512_mul_ps( cos0, rin0 );
            tmp1 = _mm512_mul_ps( cos1, rin1 );
            tmp0 = _mm512_fnmadd_ps( sin0, iin0, tmp0 );
            tmp1 = _mm512_fnmadd_ps( sin1, iin1, tmp1 );
            tmp0 = _mm512_mul_ps( scale0, tmp0 );
            tmp1 = _mm512_mul_ps( scale1, tmp1 );
            _mm512_store_ps( real_out     , tmp0 );
            _mm512_store_ps( real_out + 16, tmp1 );

            tmp0 = _mm512_mul_ps( sin0, rin0 );
            tmp1 = _mm512_mul_ps( sin1, rin1 );
            tmp0 = _mm512_fmadd_ps( cos0, iin0, tmp0 );
            tmp1 = _mm512_fmadd_ps( cos1, iin1, tmp1 );
            tmp0 = _mm512_mul_ps( scale0, tmp0 );
            tmp1 = _mm512_mul_ps( scale1, tmp1 );
            _mm512_store_ps( imag_out     , tmp0 );
            _mm512_store_ps( imag_out + 16, tmp1 );

            cos      = cos      + 32;
            sin      = sin      + 32;
            real_in  = real_in  + 32;
            real_out = real_out + 32;
            imag_in  = imag_in  + 32;
            imag_out = imag_out + 32;
        }
    }
    else
    {
        while ( k-- )
        {
            cos0 = _mm512_load_ps( cos      );
            cos1 = _mm512_load_ps( cos + 16 );
            sin0 = _mm512_load_ps( sin      );
            sin1 = _mm512_load_ps( sin + 16 );
            rin0 = _mm512_load_ps( real_in      );
            rin1 = _mm512_load_ps( real_in + 16 );
            iin0 = _mm512_load_ps( imag_in      );
            iin1 = _mm512_load_ps( imag_in + 16 );

            tmp0 = _mm512_mul_ps( cos0, rin0 );
            tmp1 = _mm512_mul_ps( cos1, rin1 );
            tmp0 = _mm512_fmadd_ps( sin0, iin0, tmp0 );
            tmp1 = _mm512_fmadd_ps( sin1, iin1, tmp1 );
            tmp0 = _mm512_mul_ps( scale0, tmp0 );
            tmp1 = _mm512_mul_ps( scale1, tmp1 );
            _mm512_store_ps( real_out     , tmp0 );
            _mm512_store_ps( real_out + 16, tmp1 );

            tmp0 = _mm512_mul_ps( sin0, rin0 );
            tmp1 = _mm512_mul_ps( sin1, rin1 );
            tmp0 = _mm512_fmsub_ps( cos0, iin0, tmp0 );
            tmp1 = _mm512_fmsub_ps( cos1, iin1, tmp1 );
            tmp0 = _mm512_mul_ps( scale0, tmp0 );
            tmp1 = _mm512_mul_ps( scale1, tmp1 );
            _mm512_store_ps( imag_out     , tmp0 );
            _mm512_store_ps( imag_out + 16, tmp1 );

            cos      = cos      + 32;
            sin      = sin      + 32;
            real_in  = real_in  + 32;
            real_out = real_out + 32;
            imag_in  = imag_in  + 32;
            imag_out = imag_out + 32;
        }
    }
}

void microkernel_float_avx512f::
swap( const float *mat, const float *in,
            float *out, size_t k, bool pattern ) const noexcept
{
    bool k_odd = k &  1;
         k     = k >> 1;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorps        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorps        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorps        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorps        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorps        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorps        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorps        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorps        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorps        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorps        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorps        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorps        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorps        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorps        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorps        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorps        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorps        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorps        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorps        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorps        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorps        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorps        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorps        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorps        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorps        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorps        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorps        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        "testq         %[k],%[k]                 \n\t"
        "jz            .Lcheckodd%=              \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovaps          (%[in]), %%zmm28       \n\t" // Even iteration.
        "vmovaps        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t" 
        "vbroadcastss    (%[mat]), %%zmm30       \n\t"
        "vbroadcastss   4(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm4  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm5  \n\t" 
        "                                        \n\t"
        "vbroadcastss   8(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  12(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm12 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm13 \n\t" 
        "                                        \n\t"
        "vbroadcastss  16(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  20(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm20 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm21 \n\t" 
        "                                        \n\t"
        "vbroadcastss  24(%[mat]), %%zmm30       \n\t"
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "                                        \n\t"
        "vmovaps       128(%[in]), %%zmm28       \n\t" // Odd iteration.
        "vmovaps       192(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastss  28(%[mat]), %%zmm30       \n\t" 
        "vbroadcastss  32(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm2  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm3  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastss  36(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  40(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm10 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm11 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastss  44(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  48(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm18 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm19 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastss  52(%[mat]), %%zmm30       \n\t"
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm26 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm27 \n\t"
        "                                        \n\t"
        "addq         $256,%[in]                 \n\t" 
        "addq          $56,%[mat]                \n\t"
        "decq          %[k]                      \n\t"
        "jnz .Lloop%=                            \n\t"
        "                                        \n\t"
        ".Lcheckodd%=:                           \n\t"
        "testb         %[k_odd],%[k_odd]         \n\t" // Remaining even iteration, if applicable.
        "jz            .Lstoreresult%=           \n\t"
        "vmovaps          (%[in]), %%zmm28       \n\t" 
        "vmovaps        64(%[in]), %%zmm29       \n\t" 
        "vbroadcastss    (%[mat]), %%zmm30       \n\t" 
        "vbroadcastss   4(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm4  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm5  \n\t" 
        "                                        \n\t"
        "vbroadcastss   8(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  12(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm12 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm13 \n\t" 
        "                                        \n\t"
        "vbroadcastss  16(%[mat]), %%zmm30       \n\t"
        "vbroadcastss  20(%[mat]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm20 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm21 \n\t" 
        "                                        \n\t"
        "vbroadcastss  24(%[mat]), %%zmm30       \n\t"
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm25 \n\t"
        "                                        \n\t"
        "                                        \n\t"
        ".Lstoreresult%=:                        \n\t" // Store the result.
        "testb        %[pattern],%[pattern]      \n\t" // Change order, if pattern is false.
        "jz           .Lnegativepattern%=        \n\t"
        "vmovaps       %%zmm0,      (%[out])     \n\t"
        "vmovaps       %%zmm1,    64(%[out])     \n\t"
        "vmovaps       %%zmm2,   128(%[out])     \n\t"
        "vmovaps       %%zmm3,   192(%[out])     \n\t"
        "vmovaps       %%zmm4,   256(%[out])     \n\t"
        "vmovaps       %%zmm5,   320(%[out])     \n\t"
        "vmovaps       %%zmm6,   384(%[out])     \n\t" 
        "vmovaps       %%zmm7,   448(%[out])     \n\t" 
        "vmovaps       %%zmm8,   512(%[out])     \n\t" 
        "vmovaps       %%zmm9,   576(%[out])     \n\t" 
        "vmovaps       %%zmm10,  640(%[out])     \n\t" 
        "vmovaps       %%zmm11,  704(%[out])     \n\t" 
        "vmovaps       %%zmm12,  768(%[out])     \n\t" 
        "vmovaps       %%zmm13,  832(%[out])     \n\t" 
        "vmovaps       %%zmm14,  896(%[out])     \n\t" 
        "vmovaps       %%zmm15,  960(%[out])     \n\t" 
        "vmovaps       %%zmm16, 1024(%[out])     \n\t" 
        "vmovaps       %%zmm17, 1088(%[out])     \n\t" 
        "vmovaps       %%zmm18, 1152(%[out])     \n\t" 
        "vmovaps       %%zmm19, 1216(%[out])     \n\t" 
        "vmovaps       %%zmm20, 1280(%[out])     \n\t" 
        "vmovaps       %%zmm21, 1344(%[out])     \n\t" 
        "vmovaps       %%zmm22, 1408(%[out])     \n\t" 
        "vmovaps       %%zmm23, 1472(%[out])     \n\t" 
        "vmovaps       %%zmm24, 1536(%[out])     \n\t" 
        "vmovaps       %%zmm25, 1600(%[out])     \n\t" 
        "vmovaps       %%zmm26, 1664(%[out])     \n\t" 
        "vmovaps       %%zmm27, 1728(%[out])     \n\t" 
        "jmp          .Lfinish%=                 \n\t"
        "                                        \n\t"
        ".Lnegativepattern%=:                    \n\t"
        "vmovaps       %%zmm2,      (%[out])     \n\t"
        "vmovaps       %%zmm3,    64(%[out])     \n\t"
        "vmovaps       %%zmm0,   128(%[out])     \n\t"
        "vmovaps       %%zmm1,   192(%[out])     \n\t"
        "vmovaps       %%zmm6,   256(%[out])     \n\t"
        "vmovaps       %%zmm7,   320(%[out])     \n\t"
        "vmovaps       %%zmm4,   384(%[out])     \n\t" 
        "vmovaps       %%zmm5,   448(%[out])     \n\t" 
        "vmovaps       %%zmm10,  512(%[out])     \n\t" 
        "vmovaps       %%zmm11,  576(%[out])     \n\t" 
        "vmovaps       %%zmm8,   640(%[out])     \n\t" 
        "vmovaps       %%zmm9,   704(%[out])     \n\t" 
        "vmovaps       %%zmm14,  768(%[out])     \n\t" 
        "vmovaps       %%zmm15,  832(%[out])     \n\t" 
        "vmovaps       %%zmm12,  896(%[out])     \n\t" 
        "vmovaps       %%zmm13,  960(%[out])     \n\t" 
        "vmovaps       %%zmm18, 1024(%[out])     \n\t" 
        "vmovaps       %%zmm19, 1088(%[out])     \n\t" 
        "vmovaps       %%zmm16, 1152(%[out])     \n\t" 
        "vmovaps       %%zmm17, 1216(%[out])     \n\t" 
        "vmovaps       %%zmm22, 1280(%[out])     \n\t" 
        "vmovaps       %%zmm23, 1344(%[out])     \n\t" 
        "vmovaps       %%zmm20, 1408(%[out])     \n\t" 
        "vmovaps       %%zmm21, 1472(%[out])     \n\t" 
        "vmovaps       %%zmm26, 1536(%[out])     \n\t" 
        "vmovaps       %%zmm27, 1600(%[out])     \n\t" 
        "vmovaps       %%zmm24, 1664(%[out])     \n\t" 
        "vmovaps       %%zmm25, 1728(%[out])     \n\t" 
        ".Lfinish%=:                             \n\t"

        : // Output operands
          [mat]     "+r"(mat),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [k_odd]   "r"(k_odd)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_float_avx512f::
zm2l( const float* fac, const float* in,
            float* out, size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;
    const float signs { -0.0 };

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorps        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorps        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorps        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorps        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorps        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorps        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorps        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorps        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorps        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorps        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorps        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorps        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorps        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorps        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorps        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorps        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorps        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorps        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorps        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorps        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorps        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorps        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorps        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorps        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorps        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorps        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorps        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovaps          (%[in]), %%zmm28       \n\t"  
        "vmovaps        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastss    (%[fac]), %%zmm30       \n\t"
        "vbroadcastss   4(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm2  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm3  \n\t" 
        "                                        \n\t"
        "vbroadcastss   8(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  12(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm4  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm5  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastss  16(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  20(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm10 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm11 \n\t" 
        "                                        \n\t"
        "vbroadcastss  24(%[fac]), %%zmm30       \n\t"
        "vbroadcastss  28(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm12 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm13 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastss  32(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  36(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm18 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm19 \n\t" 
        "                                        \n\t"
        "vbroadcastss  40(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  44(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm20 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm21 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastss  48(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  52(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm26 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm27 \n\t" 
        "                                        \n\t"
        "addq         $128,%[in]                 \n\t" 
        "addq           $4,%[fac]                \n\t" 
        "decq          %[k]                      \n\t" 
        "jnz           .Lloop%=                  \n\t" 
        "                                        \n\t"
        "vbroadcastss  %[signs], %%zmm31         \n\t" 
        "testb         %[pattern], %[pattern]    \n\t"
        "jz           .Lnegativepattern%=        \n\t"
        "vxorps        %%zmm31, %%zmm0, %%zmm0   \n\t"
        "vxorps        %%zmm31, %%zmm1, %%zmm1   \n\t"
        "vxorps        %%zmm31, %%zmm4, %%zmm4   \n\t" 
        "vxorps        %%zmm31, %%zmm5, %%zmm5   \n\t" 
        "vxorps        %%zmm31, %%zmm8, %%zmm8   \n\t" 
        "vxorps        %%zmm31, %%zmm9, %%zmm9   \n\t" 
        "vxorps        %%zmm31, %%zmm12, %%zmm12 \n\t" 
        "vxorps        %%zmm31, %%zmm13, %%zmm13 \n\t" 
        "vxorps        %%zmm31, %%zmm16, %%zmm16 \n\t" 
        "vxorps        %%zmm31, %%zmm17, %%zmm17 \n\t" 
        "vxorps        %%zmm31, %%zmm20, %%zmm20 \n\t" 
        "vxorps        %%zmm31, %%zmm21, %%zmm21 \n\t" 
        "vxorps        %%zmm31, %%zmm24, %%zmm24 \n\t" 
        "vxorps        %%zmm31, %%zmm25, %%zmm25 \n\t" 
        "jmp           .Lstoreresult%=           \n\t" 
        "                                        \n\t" 
        ".Lnegativepattern%=:                    \n\t" 
        "vxorps        %%zmm31, %%zmm2, %%zmm2   \n\t"
        "vxorps        %%zmm31, %%zmm3, %%zmm3   \n\t"
        "vxorps        %%zmm31, %%zmm6, %%zmm6   \n\t"
        "vxorps        %%zmm31, %%zmm7, %%zmm7   \n\t"
        "vxorps        %%zmm31, %%zmm10, %%zmm10 \n\t"
        "vxorps        %%zmm31, %%zmm11, %%zmm11 \n\t"
        "vxorps        %%zmm31, %%zmm14, %%zmm14 \n\t"
        "vxorps        %%zmm31, %%zmm15, %%zmm15 \n\t"
        "vxorps        %%zmm31, %%zmm18, %%zmm18 \n\t"
        "vxorps        %%zmm31, %%zmm19, %%zmm19 \n\t"
        "vxorps        %%zmm31, %%zmm22, %%zmm22 \n\t"
        "vxorps        %%zmm31, %%zmm23, %%zmm23 \n\t"
        "vxorps        %%zmm31, %%zmm26, %%zmm26 \n\t"
        "vxorps        %%zmm31, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t" 
        ".Lstoreresult%=:                        \n\t"
        "vmovaps       %%zmm0,      (%[out])     \n\t"
        "vmovaps       %%zmm1,    64(%[out])     \n\t"
        "vmovaps       %%zmm2,   128(%[out])     \n\t"
        "vmovaps       %%zmm3,   192(%[out])     \n\t"
        "vmovaps       %%zmm4,   256(%[out])     \n\t"
        "vmovaps       %%zmm5,   320(%[out])     \n\t"
        "vmovaps       %%zmm6,   384(%[out])     \n\t" 
        "vmovaps       %%zmm7,   448(%[out])     \n\t" 
        "vmovaps       %%zmm8,   512(%[out])     \n\t" 
        "vmovaps       %%zmm9,   576(%[out])     \n\t" 
        "vmovaps       %%zmm10,  640(%[out])     \n\t" 
        "vmovaps       %%zmm11,  704(%[out])     \n\t" 
        "vmovaps       %%zmm12,  768(%[out])     \n\t" 
        "vmovaps       %%zmm13,  832(%[out])     \n\t" 
        "vmovaps       %%zmm14,  896(%[out])     \n\t" 
        "vmovaps       %%zmm15,  960(%[out])     \n\t" 
        "vmovaps       %%zmm16, 1024(%[out])     \n\t" 
        "vmovaps       %%zmm17, 1088(%[out])     \n\t" 
        "vmovaps       %%zmm18, 1152(%[out])     \n\t" 
        "vmovaps       %%zmm19, 1216(%[out])     \n\t" 
        "vmovaps       %%zmm20, 1280(%[out])     \n\t" 
        "vmovaps       %%zmm21, 1344(%[out])     \n\t" 
        "vmovaps       %%zmm22, 1408(%[out])     \n\t" 
        "vmovaps       %%zmm23, 1472(%[out])     \n\t" 
        "vmovaps       %%zmm24, 1536(%[out])     \n\t" 
        "vmovaps       %%zmm25, 1600(%[out])     \n\t" 
        "vmovaps       %%zmm26, 1664(%[out])     \n\t" 
        "vmovaps       %%zmm27, 1728(%[out])     \n\t" 
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [signs]   "m"(signs)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_float_avx512f::
zm2m( const float* fac, const float* in,
            float* out, size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorps        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorps        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorps        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorps        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorps        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorps        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorps        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorps        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorps        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorps        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorps        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorps        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorps        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorps        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorps        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorps        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorps        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorps        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorps        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorps        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorps        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorps        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorps        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorps        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorps        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorps        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorps        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovaps          (%[in]), %%zmm28       \n\t"  
        "vmovaps        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastss    (%[fac]), %%zmm30       \n\t"
        "vbroadcastss   4(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm2  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm3  \n\t" 
        "                                        \n\t"
        "vbroadcastss   8(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  12(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm4  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm5  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastss  16(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  20(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm10 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm11 \n\t" 
        "                                        \n\t"
        "vbroadcastss  24(%[fac]), %%zmm30       \n\t"
        "vbroadcastss  28(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm12 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm13 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastss  32(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  36(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm18 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm19 \n\t" 
        "                                        \n\t"
        "vbroadcastss  40(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  44(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm20 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm21 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastss  48(%[fac]), %%zmm30       \n\t" 
        "vbroadcastss  52(%[fac]), %%zmm31       \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "vfmadd231ps   %%zmm28, %%zmm31, %%zmm26 \n\t" 
        "vfmadd231ps   %%zmm29, %%zmm31, %%zmm27 \n\t" 
        "                                        \n\t"
        "addq         $128,%[in]                 \n\t" 
        "addq           $4,%[fac]                \n\t" 
        "decq          %[k]                      \n\t" 
        "jnz           .Lloop%=                  \n\t" 
        "                                        \n\t" 
        "vmovaps       %%zmm26,     (%[out])     \n\t"
        "vmovaps       %%zmm27,   64(%[out])     \n\t"
        "vmovaps       %%zmm24,  128(%[out])     \n\t"
        "vmovaps       %%zmm25,  192(%[out])     \n\t"
        "vmovaps       %%zmm22,  256(%[out])     \n\t"
        "vmovaps       %%zmm23,  320(%[out])     \n\t"
        "vmovaps       %%zmm20,  384(%[out])     \n\t" 
        "vmovaps       %%zmm21,  448(%[out])     \n\t" 
        "vmovaps       %%zmm18,  512(%[out])     \n\t" 
        "vmovaps       %%zmm19,  576(%[out])     \n\t" 
        "vmovaps       %%zmm16,  640(%[out])     \n\t" 
        "vmovaps       %%zmm17,  704(%[out])     \n\t" 
        "vmovaps       %%zmm14,  768(%[out])     \n\t" 
        "vmovaps       %%zmm15,  832(%[out])     \n\t" 
        "vmovaps       %%zmm12,  896(%[out])     \n\t" 
        "vmovaps       %%zmm13,  960(%[out])     \n\t" 
        "vmovaps       %%zmm10, 1024(%[out])     \n\t" 
        "vmovaps       %%zmm11, 1088(%[out])     \n\t" 
        "vmovaps       %%zmm8,  1152(%[out])     \n\t" 
        "vmovaps       %%zmm9,  1216(%[out])     \n\t" 
        "vmovaps       %%zmm6,  1280(%[out])     \n\t" 
        "vmovaps       %%zmm7,  1344(%[out])     \n\t" 
        "vmovaps       %%zmm4,  1408(%[out])     \n\t" 
        "vmovaps       %%zmm5,  1472(%[out])     \n\t" 
        "vmovaps       %%zmm2,  1536(%[out])     \n\t" 
        "vmovaps       %%zmm3,  1600(%[out])     \n\t" 
        "vmovaps       %%zmm0,  1664(%[out])     \n\t" 
        "vmovaps       %%zmm1,  1728(%[out])     \n\t" 
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_float_avx512f::
swap2trans_buf( const float *__restrict__ real_in, 
                const float *__restrict__ imag_in,
                      float **real_out, 
                      float **imag_out,
                      size_t n ) const noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        float *__restrict__ rdst = real_out[m] + (n-m)*32;
        float *__restrict__ idst = imag_out[m] + (n-m)*32;

        __m512d rtmp0 = _mm512_load_pd( real_in      );
        __m512d rtmp1 = _mm512_load_pd( real_in + 16 );
        __m512d itmp0 = _mm512_load_pd( imag_in      );
        __m512d itmp1 = _mm512_load_pd( imag_in + 16  );
        _mm512_store_pd( rdst     , rtmp0 );
        _mm512_store_pd( rdst + 16, rtmp1 );
        _mm512_store_pd( idst     , itmp0 );
        _mm512_store_pd( idst + 16, itmp1 );
        real_in += 32;
        imag_in += 32;
    }
}

void microkernel_float_avx512f::
trans2swap_buf( const float *const *const real_in, 
                const float *const *const imag_in,
                      float *__restrict__ real_out, 
                      float *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    __m512d zeros = _mm512_setzero_pd();
    for ( size_t m = 0; m <= n; ++m )
    {
        if ( m < Pmax )
        {
            const float *__restrict__ rsrc = real_in[m] + (n-m)*32;
            const float *__restrict__ isrc = imag_in[m] + (n-m)*32;
            __m512d rtmp0 = _mm512_load_pd( rsrc      );
            __m512d rtmp1 = _mm512_load_pd( rsrc + 16 );
            __m512d itmp0 = _mm512_load_pd( isrc      );
            __m512d itmp1 = _mm512_load_pd( isrc + 16 );
            _mm512_store_pd( real_out     , rtmp0 );
            _mm512_store_pd( real_out + 16, rtmp1 );
            _mm512_store_pd( imag_out     , itmp0 );
            _mm512_store_pd( imag_out + 16, itmp1 );
        }
        else
        {
            _mm512_store_pd( real_out     , zeros );
            _mm512_store_pd( real_out + 16, zeros );
            _mm512_store_pd( imag_out     , zeros );
            _mm512_store_pd( imag_out + 16, zeros );
        }
        real_out += 32;
        imag_out += 32;
    }
}

void microkernel_float_avx512f::
buf2solid( const float   *real_in, const float *imag_in, 
                 float  **p_solids,  float *trash, const size_t *P, size_t n ) const noexcept
{
    float *solids[ 32 ] =
    {
        ( n < P[ 0] ) ? p_solids[ 0] : trash,
        ( n < P[ 1] ) ? p_solids[ 1] : trash,
        ( n < P[ 2] ) ? p_solids[ 2] : trash,
        ( n < P[ 3] ) ? p_solids[ 3] : trash,
        ( n < P[ 4] ) ? p_solids[ 4] : trash,
        ( n < P[ 5] ) ? p_solids[ 5] : trash,
        ( n < P[ 6] ) ? p_solids[ 6] : trash,
        ( n < P[ 7] ) ? p_solids[ 7] : trash,
        ( n < P[ 8] ) ? p_solids[ 8] : trash,
        ( n < P[ 9] ) ? p_solids[ 9] : trash,
        ( n < P[10] ) ? p_solids[10] : trash,
        ( n < P[11] ) ? p_solids[11] : trash,
        ( n < P[12] ) ? p_solids[12] : trash,
        ( n < P[13] ) ? p_solids[13] : trash,
        ( n < P[14] ) ? p_solids[14] : trash,
        ( n < P[15] ) ? p_solids[15] : trash,
        ( n < P[16] ) ? p_solids[16] : trash,
        ( n < P[17] ) ? p_solids[17] : trash,
        ( n < P[18] ) ? p_solids[18] : trash,
        ( n < P[19] ) ? p_solids[19] : trash,
        ( n < P[20] ) ? p_solids[20] : trash,
        ( n < P[21] ) ? p_solids[21] : trash,
        ( n < P[22] ) ? p_solids[22] : trash,
        ( n < P[23] ) ? p_solids[23] : trash,
        ( n < P[24] ) ? p_solids[24] : trash,
        ( n < P[25] ) ? p_solids[25] : trash,
        ( n < P[26] ) ? p_solids[26] : trash,
        ( n < P[27] ) ? p_solids[27] : trash,
        ( n < P[28] ) ? p_solids[28] : trash,
        ( n < P[29] ) ? p_solids[29] : trash,
        ( n < P[30] ) ? p_solids[30] : trash,
        ( n < P[31] ) ? p_solids[31] : trash
    };

    for ( size_t l = 0; l < 32; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        solids[l][ (n  )*(n+1) + m ] += real_in[ m*32 + l ];
        solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*32 + l ];
    }
}

void microkernel_float_avx512f::
solid2buf( const float *const *p_solids, const float *const zeros, const size_t *P,
                 float *real_out, float *imag_out, size_t n ) const noexcept
{
    const float *const solids[ 32 ] =
    {
        ( n < P[ 0] ) ? p_solids[ 0] : zeros,
        ( n < P[ 1] ) ? p_solids[ 1] : zeros,
        ( n < P[ 2] ) ? p_solids[ 2] : zeros,
        ( n < P[ 3] ) ? p_solids[ 3] : zeros,
        ( n < P[ 4] ) ? p_solids[ 4] : zeros,
        ( n < P[ 5] ) ? p_solids[ 5] : zeros,
        ( n < P[ 6] ) ? p_solids[ 6] : zeros,
        ( n < P[ 7] ) ? p_solids[ 7] : zeros,
        ( n < P[ 8] ) ? p_solids[ 8] : zeros,
        ( n < P[ 9] ) ? p_solids[ 9] : zeros,
        ( n < P[10] ) ? p_solids[10] : zeros,
        ( n < P[11] ) ? p_solids[11] : zeros,
        ( n < P[12] ) ? p_solids[12] : zeros,
        ( n < P[13] ) ? p_solids[13] : zeros,
        ( n < P[14] ) ? p_solids[14] : zeros,
        ( n < P[15] ) ? p_solids[15] : zeros,
        ( n < P[16] ) ? p_solids[16] : zeros,
        ( n < P[17] ) ? p_solids[17] : zeros,
        ( n < P[18] ) ? p_solids[18] : zeros,
        ( n < P[19] ) ? p_solids[19] : zeros,
        ( n < P[20] ) ? p_solids[20] : zeros,
        ( n < P[21] ) ? p_solids[21] : zeros,
        ( n < P[22] ) ? p_solids[22] : zeros,
        ( n < P[23] ) ? p_solids[23] : zeros,
        ( n < P[24] ) ? p_solids[24] : zeros,
        ( n < P[25] ) ? p_solids[25] : zeros,
        ( n < P[26] ) ? p_solids[26] : zeros,
        ( n < P[27] ) ? p_solids[27] : zeros,
        ( n < P[28] ) ? p_solids[28] : zeros,
        ( n < P[29] ) ? p_solids[29] : zeros,
        ( n < P[30] ) ? p_solids[30] : zeros,
        ( n < P[31] ) ? p_solids[31] : zeros
    };

    for ( size_t l = 0; l < 32; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        real_out[ m*32 + l ] = solids[l][ (n  )*(n+1) + m ];
        imag_out[ m*32 + l ] = solids[l][ (n+1)*(n+1) + m ];
    }
}

///////////////////////
// Double Precision. //
///////////////////////

bool microkernel_double_avx512f::available() noexcept
{
    return __builtin_cpu_supports( "avx512f" );
}

microkernel_double_avx512f::microkernel_double_avx512f(): microkernel { 14, 16, 64 }
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_double_avx512f: "
                                   "AVX-512F instructions unavailable on this CPU." };
}

void microkernel_double_avx512f::
euler( const double *x,   const double *y, const double *z,
       double *r,         double *rinv,
       double *cos_alpha, double *sin_alpha,
       double *cos_beta,  double *sin_beta,
       size_t k ) const noexcept
{
    __m512d ones  = _mm512_set1_pd(1.0);
    __m512d zeros = _mm512_setzero_pd();
    __m512d r0, r1, rinv0, rinv1; 
    __m512d ca0, sa0, ca1, sa1;   // cos/sin alpha
    __m512d cb0, sb0, cb1, sb1;   // cos/sin beta

    if ( k-- )
    {
        _mm512_store_pd( r            , ones  );
        _mm512_store_pd( r         + 8, ones  );
        _mm512_store_pd( rinv         , ones  );
        _mm512_store_pd( rinv      + 8, ones  );
        _mm512_store_pd( cos_alpha    , ones  );
        _mm512_store_pd( cos_alpha + 8, ones  );
        _mm512_store_pd( sin_alpha    , zeros );
        _mm512_store_pd( sin_alpha + 8, zeros );
        _mm512_store_pd( cos_beta     , ones  );
        _mm512_store_pd( cos_beta  + 8, ones  );
        _mm512_store_pd( sin_beta     , zeros );
        _mm512_store_pd( sin_beta  + 8, zeros );

        r         = r         + 16;
        rinv      = rinv      + 16;
        cos_alpha = cos_alpha + 16;
        sin_alpha = sin_alpha + 16;
        cos_beta  = cos_beta  + 16;
        sin_beta  = sin_beta  + 16;
    }
    else return;

    if ( k-- )
    {
        __m512d x0, x1, y0, y1, z0, z1;
        __m512d rxyinv0, rxyinv1, rxy0, rxy1;

        x0   = _mm512_load_pd(x+0); 
        x1   = _mm512_load_pd(x+8); 
        y0   = _mm512_load_pd(y+0); 
        y1   = _mm512_load_pd(y+8); 
        z0   = _mm512_load_pd(z+0); 
        z1   = _mm512_load_pd(z+8); 

        rxy0 = _mm512_mul_pd(x0,x0);
        rxy1 = _mm512_mul_pd(x1,x1);
        rxy0 = _mm512_fmadd_pd(y0,y0,rxy0);
        rxy1 = _mm512_fmadd_pd(y1,y1,rxy1);
        r0   = _mm512_fmadd_pd(z0,z0,rxy0);
        r1   = _mm512_fmadd_pd(z1,z1,rxy1);

        rxy0 = _mm512_sqrt_pd( rxy0 );
        rxy1 = _mm512_sqrt_pd( rxy1 );
        r0   = _mm512_sqrt_pd( r0 );
        r1   = _mm512_sqrt_pd( r1 );

        rxyinv0 = _mm512_div_pd( ones, rxy0 );
        rxyinv1 = _mm512_div_pd( ones, rxy1 );
        rinv0   = _mm512_div_pd( ones, r0 );
        rinv1   = _mm512_div_pd( ones, r1 );

        ca0 = _mm512_mul_pd( y0, rxyinv0 );
        ca1 = _mm512_mul_pd( y1, rxyinv1 );
        sa0 = _mm512_mul_pd( x0, rxyinv0 );
        sa1 = _mm512_mul_pd( x1, rxyinv1 );

        __mmask8 iszero0 = _mm512_cmp_pd_mask( rxy0, zeros, _CMP_EQ_OQ );
        __mmask8 iszero1 = _mm512_cmp_pd_mask( rxy1, zeros, _CMP_EQ_OQ );
        ca0 = _mm512_mask_mov_pd( ca0, iszero0, ones );
        ca1 = _mm512_mask_mov_pd( ca1, iszero1, ones );
        sa0 = _mm512_mask_mov_pd( sa0, iszero0, zeros );
        sa1 = _mm512_mask_mov_pd( sa1, iszero1, zeros );

        cb0 = _mm512_mul_pd( z0, rinv0 );
        cb1 = _mm512_mul_pd( z1, rinv1 );
        sb0 = _mm512_fnmadd_pd( rxy0, rinv0, zeros );
        sb1 = _mm512_fnmadd_pd( rxy1, rinv1, zeros );

        _mm512_store_pd( r            , r0 );
        _mm512_store_pd( r         + 8, r1 );
        _mm512_store_pd( rinv         , rinv0 );
        _mm512_store_pd( rinv      + 8, rinv1 );
        _mm512_store_pd( cos_alpha    , ca0 );
        _mm512_store_pd( cos_alpha + 8, ca1 );
        _mm512_store_pd( sin_alpha    , sa0 );
        _mm512_store_pd( sin_alpha + 8, sa1 );
        _mm512_store_pd( cos_beta     , cb0 );
        _mm512_store_pd( cos_beta  + 8, cb1 );
        _mm512_store_pd( sin_beta     , sb0 );
        _mm512_store_pd( sin_beta  + 8, sb1 );
    }
    else return;

    while ( k-- )
    {
        __m512d rn0, rn1, rninv0, rninv1;
        __m512d cn0, sn0, cn1, sn1, res0, res1;
       
        // r^{n+1}    = r^{ n} * r; 
        // r^{-(n+1)} = r^{-n} * r^{-1}; 
        rn0    = _mm512_load_pd( r     ); 
        rn1    = _mm512_load_pd( r + 8 ); 
        rninv0 = _mm512_load_pd( rinv     ); 
        rninv1 = _mm512_load_pd( rinv + 8 ); 
        
        rn0    = _mm512_mul_pd( rn0, r0 );
        rn1    = _mm512_mul_pd( rn1, r1 );
        rninv0 = _mm512_mul_pd( rninv0, rinv0 );
        rninv1 = _mm512_mul_pd( rninv1, rinv1 );
        _mm512_store_pd( r    + 16, rn0 );
        _mm512_store_pd( r    + 24, rn1 );
        _mm512_store_pd( rinv + 16, rninv0 );
        _mm512_store_pd( rinv + 24, rninv1 );

        // cos(alpha*(n+1)) = cos(alpha*n)*cos(alpha) - sin(alpha*n)*sin(alpha);
        // sin(alpha*(n+1)) = sin(alpha*n)*cos(alpha) + cos(alpha*n)*sin(alpha);
        cn0  = _mm512_load_pd( cos_alpha     );
        cn1  = _mm512_load_pd( cos_alpha + 8 );
        sn0  = _mm512_load_pd( sin_alpha     );
        sn1  = _mm512_load_pd( sin_alpha + 8 );
        res0 = _mm512_mul_pd(cn0,ca0);
        res1 = _mm512_mul_pd(cn1,ca1);
        res0 = _mm512_fnmadd_pd(sn0,sa0,res0);
        res1 = _mm512_fnmadd_pd(sn1,sa1,res1);
        _mm512_store_pd( cos_alpha + 16, res0 );
        _mm512_store_pd( cos_alpha + 24, res1 );
        res0 = _mm512_mul_pd(sn0,ca0);
        res1 = _mm512_mul_pd(sn1,ca1);
        res0 = _mm512_fmadd_pd(cn0,sa0,res0);
        res1 = _mm512_fmadd_pd(cn1,sa1,res1);
        _mm512_store_pd( sin_alpha + 16, res0 );
        _mm512_store_pd( sin_alpha + 24, res1 );

        // cos(beta*(n+1)) = cos(beta*n)*cos(beta) - sin(beta*n)*sin(beta);
        // sin(beta*(n+1)) = sin(beta*n)*cos(beta) + cos(beta*n)*sin(beta);
        cn0  = _mm512_load_pd( cos_beta     );
        cn1  = _mm512_load_pd( cos_beta + 8 );
        sn0  = _mm512_load_pd( sin_beta     );
        sn1  = _mm512_load_pd( sin_beta + 8 );
        res0 = _mm512_mul_pd(cn0,cb0);
        res1 = _mm512_mul_pd(cn1,cb1);
        res0 = _mm512_fnmadd_pd(sn0,sb0,res0);
        res1 = _mm512_fnmadd_pd(sn1,sb1,res1);
        _mm512_store_pd( cos_beta + 16, res0 );
        _mm512_store_pd( cos_beta + 24, res1 );
        res0 = _mm512_mul_pd(sn0,cb0);
        res1 = _mm512_mul_pd(sn1,cb1);
        res0 = _mm512_fmadd_pd(cn0,sb0,res0);
        res1 = _mm512_fmadd_pd(cn1,sb1,res1);
        _mm512_store_pd( sin_beta + 16, res0 );
        _mm512_store_pd( sin_beta + 24, res1 );
        
        r         = r         + 16;
        rinv      = rinv      + 16;
        cos_alpha = cos_alpha + 16;
        sin_alpha = sin_alpha + 16;
        cos_beta  = cos_beta  + 16;
        sin_beta  = sin_beta  + 16;
    }
}

void microkernel_double_avx512f::
rotscale( const double *cos,      const double *sin,      const double *scale,
          const double *real_in,  const double *imag_in,
                double *__restrict__ real_out,      
                double *__restrict__ imag_out,
          size_t k, bool forward ) const noexcept
{
    __m512d scale0, scale1;
    __m512d cos0, cos1, sin0, sin1;
    __m512d rin0, rin1;
    __m512d iin0, iin1;
    __m512d tmp0, tmp1;

    scale0 = _mm512_load_pd( scale     );
    scale1 = _mm512_load_pd( scale + 8 );

    if ( forward )
    {
        while ( k-- )
        {
            cos0 = _mm512_load_pd( cos     );
            cos1 = _mm512_load_pd( cos + 8 );
            sin0 = _mm512_load_pd( sin     );
            sin1 = _mm512_load_pd( sin + 8 );
            rin0 = _mm512_load_pd( real_in     );
            rin1 = _mm512_load_pd( real_in + 8 );
            iin0 = _mm512_load_pd( imag_in     );
            iin1 = _mm512_load_pd( imag_in + 8 );

            tmp0 = _mm512_mul_pd( cos0, rin0 );
            tmp1 = _mm512_mul_pd( cos1, rin1 );
            tmp0 = _mm512_fnmadd_pd( sin0, iin0, tmp0 );
            tmp1 = _mm512_fnmadd_pd( sin1, iin1, tmp1 );
            tmp0 = _mm512_mul_pd( scale0, tmp0 );
            tmp1 = _mm512_mul_pd( scale1, tmp1 );
            _mm512_store_pd( real_out    , tmp0 );
            _mm512_store_pd( real_out + 8, tmp1 );

            tmp0 = _mm512_mul_pd( sin0, rin0 );
            tmp1 = _mm512_mul_pd( sin1, rin1 );
            tmp0 = _mm512_fmadd_pd( cos0, iin0, tmp0 );
            tmp1 = _mm512_fmadd_pd( cos1, iin1, tmp1 );
            tmp0 = _mm512_mul_pd( scale0, tmp0 );
            tmp1 = _mm512_mul_pd( scale1, tmp1 );
            _mm512_store_pd( imag_out    , tmp0 );
            _mm512_store_pd( imag_out + 8, tmp1 );

            cos      = cos      + 16;
            sin      = sin      + 16;
            real_in  = real_in  + 16;
            real_out = real_out + 16;
            imag_in  = imag_in  + 16;
            imag_out = imag_out + 16;
        }
    }
    else
    {
        while ( k-- )
        {
            cos0 = _mm512_load_pd( cos     );
            cos1 = _mm512_load_pd( cos + 8 );
            sin0 = _mm512_load_pd( sin     );
            sin1 = _mm512_load_pd( sin + 8 );
            rin0 = _mm512_load_pd( real_in     );
            rin1 = _mm512_load_pd( real_in + 8 );
            iin0 = _mm512_load_pd( imag_in     );
            iin1 = _mm512_load_pd( imag_in + 8 );

            tmp0 = _mm512_mul_pd( cos0, rin0 );
            tmp1 = _mm512_mul_pd( cos1, rin1 );
            tmp0 = _mm512_fmadd_pd( sin0, iin0, tmp0 );
            tmp1 = _mm512_fmadd_pd( sin1, iin1, tmp1 );
            tmp0 = _mm512_mul_pd( scale0, tmp0 );
            tmp1 = _mm512_mul_pd( scale1, tmp1 );
            _mm512_store_pd( real_out    , tmp0 );
            _mm512_store_pd( real_out + 8, tmp1 );

            tmp0 = _mm512_mul_pd( sin0, rin0 );
            tmp1 = _mm512_mul_pd( sin1, rin1 );
            tmp0 = _mm512_fmsub_pd( cos0, iin0, tmp0 );
            tmp1 = _mm512_fmsub_pd( cos1, iin1, tmp1 );
            tmp0 = _mm512_mul_pd( scale0, tmp0 );
            tmp1 = _mm512_mul_pd( scale1, tmp1 );
            _mm512_store_pd( imag_out    , tmp0 );
            _mm512_store_pd( imag_out + 8, tmp1 );

            cos      = cos      + 16;
            sin      = sin      + 16;
            real_in  = real_in  + 16;
            real_out = real_out + 16;
            imag_in  = imag_in  + 16;
            imag_out = imag_out + 16;
        }
    }
}

void microkernel_double_avx512f::
swap( const double *mat, const double *in,
            double *out, size_t k, bool pattern ) const noexcept
{
    bool k_odd = k &  1;
         k     = k >> 1;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorpd        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorpd        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorpd        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorpd        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorpd        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorpd        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorpd        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorpd        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorpd        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorpd        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorpd        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorpd        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorpd        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorpd        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorpd        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorpd        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorpd        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorpd        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorpd        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorpd        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorpd        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorpd        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorpd        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorpd        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorpd        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorpd        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorpd        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        "testq         %[k],%[k]                 \n\t"
        "jz            .Lcheckodd%=              \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovapd          (%[in]), %%zmm28       \n\t" // Even iteration.
        "vmovapd        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t" 
        "vbroadcastsd    (%[mat]), %%zmm30       \n\t"
        "vbroadcastsd   8(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm4  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm5  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  16(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  24(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm12 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm13 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  32(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  40(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm20 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm21 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  48(%[mat]), %%zmm30       \n\t"
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "                                        \n\t"
        "vmovapd       128(%[in]), %%zmm28       \n\t" // Odd iteration.
        "vmovapd       192(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastsd  56(%[mat]), %%zmm30       \n\t" 
        "vbroadcastsd  64(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm2  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm3  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  72(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  80(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm10 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm11 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  88(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  96(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm18 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm19 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  104(%[mat]), %%zmm30      \n\t"
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm26 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm27 \n\t"
        "                                        \n\t"
        "addq          $256,%[in]                \n\t" 
        "addq          $112,%[mat]               \n\t"
        "decq          %[k]                      \n\t"
        "jnz .Lloop%=                            \n\t"
        "                                        \n\t"
        ".Lcheckodd%=:                           \n\t"
        "testb         %[k_odd],%[k_odd]         \n\t" // Remaining even iteration, if applicable.
        "jz            .Lstoreresult%=           \n\t"
        "vmovapd          (%[in]), %%zmm28       \n\t" 
        "vmovapd        64(%[in]), %%zmm29       \n\t" 
        "vbroadcastsd    (%[mat]), %%zmm30       \n\t" 
        "vbroadcastsd   8(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm4  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm5  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  16(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  24(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm12 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm13 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  32(%[mat]), %%zmm30       \n\t"
        "vbroadcastsd  40(%[mat]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm20 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm21 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  48(%[mat]), %%zmm30       \n\t"
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm25 \n\t"
        "                                        \n\t"
        "                                        \n\t"
        ".Lstoreresult%=:                        \n\t" // Store the result.
        "testb        %[pattern],%[pattern]      \n\t" // Change order, if pattern is false.
        "jz           .Lnegativepattern%=        \n\t"
        "vmovapd       %%zmm0,      (%[out])     \n\t"
        "vmovapd       %%zmm1,    64(%[out])     \n\t"
        "vmovapd       %%zmm2,   128(%[out])     \n\t"
        "vmovapd       %%zmm3,   192(%[out])     \n\t"
        "vmovapd       %%zmm4,   256(%[out])     \n\t"
        "vmovapd       %%zmm5,   320(%[out])     \n\t"
        "vmovapd       %%zmm6,   384(%[out])     \n\t" 
        "vmovapd       %%zmm7,   448(%[out])     \n\t" 
        "vmovapd       %%zmm8,   512(%[out])     \n\t" 
        "vmovapd       %%zmm9,   576(%[out])     \n\t" 
        "vmovapd       %%zmm10,  640(%[out])     \n\t" 
        "vmovapd       %%zmm11,  704(%[out])     \n\t" 
        "vmovapd       %%zmm12,  768(%[out])     \n\t" 
        "vmovapd       %%zmm13,  832(%[out])     \n\t" 
        "vmovapd       %%zmm14,  896(%[out])     \n\t" 
        "vmovapd       %%zmm15,  960(%[out])     \n\t" 
        "vmovapd       %%zmm16, 1024(%[out])     \n\t" 
        "vmovapd       %%zmm17, 1088(%[out])     \n\t" 
        "vmovapd       %%zmm18, 1152(%[out])     \n\t" 
        "vmovapd       %%zmm19, 1216(%[out])     \n\t" 
        "vmovapd       %%zmm20, 1280(%[out])     \n\t" 
        "vmovapd       %%zmm21, 1344(%[out])     \n\t" 
        "vmovapd       %%zmm22, 1408(%[out])     \n\t" 
        "vmovapd       %%zmm23, 1472(%[out])     \n\t" 
        "vmovapd       %%zmm24, 1536(%[out])     \n\t" 
        "vmovapd       %%zmm25, 1600(%[out])     \n\t" 
        "vmovapd       %%zmm26, 1664(%[out])     \n\t" 
        "vmovapd       %%zmm27, 1728(%[out])     \n\t" 
        "jmp          .Lfinish%=                 \n\t"
        "                                        \n\t"
        ".Lnegativepattern%=:                    \n\t"
        "vmovapd       %%zmm2,      (%[out])     \n\t"
        "vmovapd       %%zmm3,    64(%[out])     \n\t"
        "vmovapd       %%zmm0,   128(%[out])     \n\t"
        "vmovapd       %%zmm1,   192(%[out])     \n\t"
        "vmovapd       %%zmm6,   256(%[out])     \n\t"
        "vmovapd       %%zmm7,   320(%[out])     \n\t"
        "vmovapd       %%zmm4,   384(%[out])     \n\t" 
        "vmovapd       %%zmm5,   448(%[out])     \n\t" 
        "vmovapd       %%zmm10,  512(%[out])     \n\t" 
        "vmovapd       %%zmm11,  576(%[out])     \n\t" 
        "vmovapd       %%zmm8,   640(%[out])     \n\t" 
        "vmovapd       %%zmm9,   704(%[out])     \n\t" 
        "vmovapd       %%zmm14,  768(%[out])     \n\t" 
        "vmovapd       %%zmm15,  832(%[out])     \n\t" 
        "vmovapd       %%zmm12,  896(%[out])     \n\t" 
        "vmovapd       %%zmm13,  960(%[out])     \n\t" 
        "vmovapd       %%zmm18, 1024(%[out])     \n\t" 
        "vmovapd       %%zmm19, 1088(%[out])     \n\t" 
        "vmovapd       %%zmm16, 1152(%[out])     \n\t" 
        "vmovapd       %%zmm17, 1216(%[out])     \n\t" 
        "vmovapd       %%zmm22, 1280(%[out])     \n\t" 
        "vmovapd       %%zmm23, 1344(%[out])     \n\t" 
        "vmovapd       %%zmm20, 1408(%[out])     \n\t" 
        "vmovapd       %%zmm21, 1472(%[out])     \n\t" 
        "vmovapd       %%zmm26, 1536(%[out])     \n\t" 
        "vmovapd       %%zmm27, 1600(%[out])     \n\t" 
        "vmovapd       %%zmm24, 1664(%[out])     \n\t" 
        "vmovapd       %%zmm25, 1728(%[out])     \n\t" 
        ".Lfinish%=:                             \n\t"

        : // Output operands
          [mat]     "+r"(mat),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [k_odd]   "r"(k_odd)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_double_avx512f::
zm2l( const double* fac, const double* in,
            double* out, size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;
    const double signs { -0.0 };

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorpd        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorpd        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorpd        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorpd        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorpd        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorpd        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorpd        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorpd        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorpd        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorpd        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorpd        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorpd        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorpd        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorpd        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorpd        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorpd        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorpd        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorpd        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorpd        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorpd        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorpd        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorpd        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorpd        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorpd        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorpd        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorpd        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorpd        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovapd          (%[in]), %%zmm28       \n\t"  
        "vmovapd        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastsd    (%[fac]), %%zmm30       \n\t"
        "vbroadcastsd   8(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm2  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm3  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  16(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  24(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm4  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm5  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  32(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  40(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm10 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm11 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  48(%[fac]), %%zmm30       \n\t"
        "vbroadcastsd  56(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm12 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm13 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  64(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  72(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm18 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm19 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  80(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  88(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm20 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm21 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastsd   96(%[fac]), %%zmm30      \n\t" 
        "vbroadcastsd  104(%[fac]), %%zmm31      \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm26 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm27 \n\t" 
        "                                        \n\t"
        "addq         $128,%[in]                 \n\t" 
        "addq           $8,%[fac]                \n\t" 
        "decq          %[k]                      \n\t" 
        "jnz           .Lloop%=                  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  %[signs], %%zmm31         \n\t" 
        "testb         %[pattern], %[pattern]    \n\t"
        "jz           .Lnegativepattern%=        \n\t"
        "vxorpd        %%zmm31, %%zmm0, %%zmm0   \n\t"
        "vxorpd        %%zmm31, %%zmm1, %%zmm1   \n\t"
        "vxorpd        %%zmm31, %%zmm4, %%zmm4   \n\t" 
        "vxorpd        %%zmm31, %%zmm5, %%zmm5   \n\t" 
        "vxorpd        %%zmm31, %%zmm8, %%zmm8   \n\t" 
        "vxorpd        %%zmm31, %%zmm9, %%zmm9   \n\t" 
        "vxorpd        %%zmm31, %%zmm12, %%zmm12 \n\t" 
        "vxorpd        %%zmm31, %%zmm13, %%zmm13 \n\t" 
        "vxorpd        %%zmm31, %%zmm16, %%zmm16 \n\t" 
        "vxorpd        %%zmm31, %%zmm17, %%zmm17 \n\t" 
        "vxorpd        %%zmm31, %%zmm20, %%zmm20 \n\t" 
        "vxorpd        %%zmm31, %%zmm21, %%zmm21 \n\t" 
        "vxorpd        %%zmm31, %%zmm24, %%zmm24 \n\t" 
        "vxorpd        %%zmm31, %%zmm25, %%zmm25 \n\t" 
        "jmp           .Lstoreresult%=           \n\t" 
        "                                        \n\t" 
        ".Lnegativepattern%=:                    \n\t" 
        "vxorpd        %%zmm31, %%zmm2, %%zmm2   \n\t"
        "vxorpd        %%zmm31, %%zmm3, %%zmm3   \n\t"
        "vxorpd        %%zmm31, %%zmm6, %%zmm6   \n\t"
        "vxorpd        %%zmm31, %%zmm7, %%zmm7   \n\t"
        "vxorpd        %%zmm31, %%zmm10, %%zmm10 \n\t"
        "vxorpd        %%zmm31, %%zmm11, %%zmm11 \n\t"
        "vxorpd        %%zmm31, %%zmm14, %%zmm14 \n\t"
        "vxorpd        %%zmm31, %%zmm15, %%zmm15 \n\t"
        "vxorpd        %%zmm31, %%zmm18, %%zmm18 \n\t"
        "vxorpd        %%zmm31, %%zmm19, %%zmm19 \n\t"
        "vxorpd        %%zmm31, %%zmm22, %%zmm22 \n\t"
        "vxorpd        %%zmm31, %%zmm23, %%zmm23 \n\t"
        "vxorpd        %%zmm31, %%zmm26, %%zmm26 \n\t"
        "vxorpd        %%zmm31, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t" 
        ".Lstoreresult%=:                        \n\t"
        "vmovapd       %%zmm0,      (%[out])     \n\t"
        "vmovapd       %%zmm1,    64(%[out])     \n\t"
        "vmovapd       %%zmm2,   128(%[out])     \n\t"
        "vmovapd       %%zmm3,   192(%[out])     \n\t"
        "vmovapd       %%zmm4,   256(%[out])     \n\t"
        "vmovapd       %%zmm5,   320(%[out])     \n\t"
        "vmovapd       %%zmm6,   384(%[out])     \n\t" 
        "vmovapd       %%zmm7,   448(%[out])     \n\t" 
        "vmovapd       %%zmm8,   512(%[out])     \n\t" 
        "vmovapd       %%zmm9,   576(%[out])     \n\t" 
        "vmovapd       %%zmm10,  640(%[out])     \n\t" 
        "vmovapd       %%zmm11,  704(%[out])     \n\t" 
        "vmovapd       %%zmm12,  768(%[out])     \n\t" 
        "vmovapd       %%zmm13,  832(%[out])     \n\t" 
        "vmovapd       %%zmm14,  896(%[out])     \n\t" 
        "vmovapd       %%zmm15,  960(%[out])     \n\t" 
        "vmovapd       %%zmm16, 1024(%[out])     \n\t" 
        "vmovapd       %%zmm17, 1088(%[out])     \n\t" 
        "vmovapd       %%zmm18, 1152(%[out])     \n\t" 
        "vmovapd       %%zmm19, 1216(%[out])     \n\t" 
        "vmovapd       %%zmm20, 1280(%[out])     \n\t" 
        "vmovapd       %%zmm21, 1344(%[out])     \n\t" 
        "vmovapd       %%zmm22, 1408(%[out])     \n\t" 
        "vmovapd       %%zmm23, 1472(%[out])     \n\t" 
        "vmovapd       %%zmm24, 1536(%[out])     \n\t" 
        "vmovapd       %%zmm25, 1600(%[out])     \n\t" 
        "vmovapd       %%zmm26, 1664(%[out])     \n\t" 
        "vmovapd       %%zmm27, 1728(%[out])     \n\t" 
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [signs]   "m"(signs)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_double_avx512f::
zm2m( const double* fac, const double* in,
            double* out, size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd        %%zmm0 , %%zmm0 , %%zmm0  \n\t"
        "vxorpd        %%zmm1 , %%zmm1 , %%zmm1  \n\t"
        "vxorpd        %%zmm2 , %%zmm2 , %%zmm2  \n\t"
        "vxorpd        %%zmm3 , %%zmm3 , %%zmm3  \n\t"
        "vxorpd        %%zmm4 , %%zmm4 , %%zmm4  \n\t"
        "vxorpd        %%zmm5 , %%zmm5 , %%zmm5  \n\t"
        "vxorpd        %%zmm6 , %%zmm6 , %%zmm6  \n\t"
        "vxorpd        %%zmm7 , %%zmm7 , %%zmm7  \n\t"
        "vxorpd        %%zmm8 , %%zmm8 , %%zmm8  \n\t"
        "vxorpd        %%zmm9 , %%zmm9 , %%zmm9  \n\t"
        "vxorpd        %%zmm10, %%zmm10, %%zmm10 \n\t"
        "vxorpd        %%zmm11, %%zmm11, %%zmm11 \n\t"
        "vxorpd        %%zmm12, %%zmm12, %%zmm12 \n\t"
        "vxorpd        %%zmm13, %%zmm13, %%zmm13 \n\t"
        "vxorpd        %%zmm14, %%zmm14, %%zmm14 \n\t"
        "vxorpd        %%zmm15, %%zmm15, %%zmm15 \n\t"
        "vxorpd        %%zmm16, %%zmm16, %%zmm16 \n\t"
        "vxorpd        %%zmm17, %%zmm17, %%zmm17 \n\t"
        "vxorpd        %%zmm18, %%zmm18, %%zmm18 \n\t"
        "vxorpd        %%zmm19, %%zmm19, %%zmm19 \n\t"
        "vxorpd        %%zmm20, %%zmm20, %%zmm20 \n\t"
        "vxorpd        %%zmm21, %%zmm21, %%zmm21 \n\t"
        "vxorpd        %%zmm22, %%zmm22, %%zmm22 \n\t"
        "vxorpd        %%zmm23, %%zmm23, %%zmm23 \n\t"
        "vxorpd        %%zmm24, %%zmm24, %%zmm24 \n\t"
        "vxorpd        %%zmm25, %%zmm25, %%zmm25 \n\t"
        "vxorpd        %%zmm26, %%zmm26, %%zmm26 \n\t"
        "vxorpd        %%zmm27, %%zmm27, %%zmm27 \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" 
        "                                        \n\t" 
        "vmovapd          (%[in]), %%zmm28       \n\t"  
        "vmovapd        64(%[in]), %%zmm29       \n\t" 
        "                                        \n\t"
        "vbroadcastsd    (%[fac]), %%zmm30       \n\t"
        "vbroadcastsd   8(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm0  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm1  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm2  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm3  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  16(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  24(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm4  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm5  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm6  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm7  \n\t" 
        "                                        \n\t"
        "vbroadcastsd  32(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  40(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm8  \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm9  \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm10 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm11 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  48(%[fac]), %%zmm30       \n\t"
        "vbroadcastsd  56(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm12 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm13 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm14 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm15 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  64(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  72(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm16 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm17 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm18 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm19 \n\t" 
        "                                        \n\t"
        "vbroadcastsd  80(%[fac]), %%zmm30       \n\t" 
        "vbroadcastsd  88(%[fac]), %%zmm31       \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm20 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm21 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm22 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm23 \n\t" 
        "                                        \n\t"
        "vbroadcastsd   96(%[fac]), %%zmm30      \n\t" 
        "vbroadcastsd  104(%[fac]), %%zmm31      \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm30, %%zmm24 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm30, %%zmm25 \n\t" 
        "vfmadd231pd   %%zmm28, %%zmm31, %%zmm26 \n\t" 
        "vfmadd231pd   %%zmm29, %%zmm31, %%zmm27 \n\t" 
        "                                        \n\t"
        "addq         $128,%[in]                 \n\t" 
        "addq           $8,%[fac]                \n\t" 
        "decq          %[k]                      \n\t" 
        "jnz           .Lloop%=                  \n\t" 
        "                                        \n\t" 
        "vmovapd       %%zmm26,     (%[out])     \n\t"
        "vmovapd       %%zmm27,   64(%[out])     \n\t"
        "vmovapd       %%zmm24,  128(%[out])     \n\t"
        "vmovapd       %%zmm25,  192(%[out])     \n\t"
        "vmovapd       %%zmm22,  256(%[out])     \n\t"
        "vmovapd       %%zmm23,  320(%[out])     \n\t"
        "vmovapd       %%zmm20,  384(%[out])     \n\t" 
        "vmovapd       %%zmm21,  448(%[out])     \n\t" 
        "vmovapd       %%zmm18,  512(%[out])     \n\t" 
        "vmovapd       %%zmm19,  576(%[out])     \n\t" 
        "vmovapd       %%zmm16,  640(%[out])     \n\t" 
        "vmovapd       %%zmm17,  704(%[out])     \n\t" 
        "vmovapd       %%zmm14,  768(%[out])     \n\t" 
        "vmovapd       %%zmm15,  832(%[out])     \n\t" 
        "vmovapd       %%zmm12,  896(%[out])     \n\t" 
        "vmovapd       %%zmm13,  960(%[out])     \n\t" 
        "vmovapd       %%zmm10, 1024(%[out])     \n\t" 
        "vmovapd       %%zmm11, 1088(%[out])     \n\t" 
        "vmovapd       %%zmm8,  1152(%[out])     \n\t" 
        "vmovapd       %%zmm9,  1216(%[out])     \n\t" 
        "vmovapd       %%zmm6,  1280(%[out])     \n\t" 
        "vmovapd       %%zmm7,  1344(%[out])     \n\t" 
        "vmovapd       %%zmm4,  1408(%[out])     \n\t" 
        "vmovapd       %%zmm5,  1472(%[out])     \n\t" 
        "vmovapd       %%zmm2,  1536(%[out])     \n\t" 
        "vmovapd       %%zmm3,  1600(%[out])     \n\t" 
        "vmovapd       %%zmm0,  1664(%[out])     \n\t" 
        "vmovapd       %%zmm1,  1728(%[out])     \n\t" 
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out)

        : // Clobbered registers
          "zmm0" , "zmm1" , "zmm2" , "zmm3" ,
          "zmm4" , "zmm5" , "zmm6" , "zmm7" ,
          "zmm8" , "zmm9" , "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "zmm16", "zmm17", "zmm18", "zmm19",
          "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27",
          "zmm28", "zmm29", "zmm30", "zmm31"
    );
}

void microkernel_double_avx512f::
swap2trans_buf( const double *__restrict__ real_in, 
                const double *__restrict__ imag_in,
                      double **real_out, 
                      double **imag_out,
                      size_t n ) const noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        double *__restrict__ rdst = real_out[m] + (n-m)*16;
        double *__restrict__ idst = imag_out[m] + (n-m)*16;

        __m512d rtmp0 = _mm512_load_pd( real_in     );
        __m512d rtmp1 = _mm512_load_pd( real_in + 8 );
        __m512d itmp0 = _mm512_load_pd( imag_in     );
        __m512d itmp1 = _mm512_load_pd( imag_in + 8 );
        _mm512_store_pd( rdst    , rtmp0 );
        _mm512_store_pd( rdst + 8, rtmp1 );
        _mm512_store_pd( idst    , itmp0 );
        _mm512_store_pd( idst + 8, itmp1 );
        real_in += 16;
        imag_in += 16;
    }
}

void microkernel_double_avx512f::
trans2swap_buf( const double *const *const real_in, 
                const double *const *const imag_in,
                      double *__restrict__ real_out, 
                      double *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    __m512d zeros = _mm512_setzero_pd();
    for ( size_t m = 0; m <= n; ++m )
    {
        if ( m < Pmax )
        {
            const double *__restrict__ rsrc = real_in[m] + (n-m)*16;
            const double *__restrict__ isrc = imag_in[m] + (n-m)*16;
            __m512d rtmp0 = _mm512_load_pd( rsrc     );
            __m512d rtmp1 = _mm512_load_pd( rsrc + 8 );
            __m512d itmp0 = _mm512_load_pd( isrc     );
            __m512d itmp1 = _mm512_load_pd( isrc + 8 );
            _mm512_store_pd( real_out    , rtmp0 );
            _mm512_store_pd( real_out + 8, rtmp1 );
            _mm512_store_pd( imag_out    , itmp0 );
            _mm512_store_pd( imag_out + 8, itmp1 );
        }
        else
        {
            _mm512_store_pd( real_out    , zeros );
            _mm512_store_pd( real_out + 8, zeros );
            _mm512_store_pd( imag_out    , zeros );
            _mm512_store_pd( imag_out + 8, zeros );
        }
        real_out += 16;
        imag_out += 16;
    }
}

#define transpose(row0,row1,row2,row3)               \
{                                                    \
    __m256d tmp3, tmp2, tmp1, tmp0;                  \
                                                     \
    tmp0 = _mm256_unpacklo_pd((row0),(row1));        \
    tmp2 = _mm256_unpackhi_pd((row0),(row1));        \
    tmp1 = _mm256_unpacklo_pd((row2),(row3));        \
    tmp3 = _mm256_unpackhi_pd((row2),(row3));        \
                                                     \
    (row0) = _mm256_permute2f128_pd(tmp0,tmp1,0x20); \
    (row1) = _mm256_permute2f128_pd(tmp2,tmp3,0x20); \
    (row2) = _mm256_permute2f128_pd(tmp0,tmp1,0x31); \
    (row3) = _mm256_permute2f128_pd(tmp2,tmp3,0x31); \
}

void microkernel_double_avx512f::
buf2solid( const double   *real_in, const double *imag_in, 
                 double  **p_solids,  double *trash, const size_t *P, size_t n ) const noexcept
{
    double *solids[ 16 ] =
    {
        ( n < P[ 0] ) ? p_solids[ 0] : trash,
        ( n < P[ 1] ) ? p_solids[ 1] : trash,
        ( n < P[ 2] ) ? p_solids[ 2] : trash,
        ( n < P[ 3] ) ? p_solids[ 3] : trash,
        ( n < P[ 4] ) ? p_solids[ 4] : trash,
        ( n < P[ 5] ) ? p_solids[ 5] : trash,
        ( n < P[ 6] ) ? p_solids[ 6] : trash,
        ( n < P[ 7] ) ? p_solids[ 7] : trash,
        ( n < P[ 8] ) ? p_solids[ 8] : trash,
        ( n < P[ 9] ) ? p_solids[ 9] : trash,
        ( n < P[10] ) ? p_solids[10] : trash,
        ( n < P[11] ) ? p_solids[11] : trash,
        ( n < P[12] ) ? p_solids[12] : trash,
        ( n < P[13] ) ? p_solids[13] : trash,
        ( n < P[14] ) ? p_solids[14] : trash,
        ( n < P[15] ) ? p_solids[15] : trash
    };

    size_t m;
    __m256d r00, r01, r10, r11, r20, r21, r30, r31;
    for ( m = 0; m + 3 <= n; m += 4 )
    {
        r00 = _mm256_load_pd( real_in + (m+0)*16 + 0 );
        r01 = _mm256_load_pd( real_in + (m+0)*16 + 4 );
        r10 = _mm256_load_pd( real_in + (m+1)*16 + 0 );
        r11 = _mm256_load_pd( real_in + (m+1)*16 + 4 );
        r20 = _mm256_load_pd( real_in + (m+2)*16 + 0 );
        r21 = _mm256_load_pd( real_in + (m+2)*16 + 4 );
        r30 = _mm256_load_pd( real_in + (m+3)*16 + 0 );
        r31 = _mm256_load_pd( real_in + (m+3)*16 + 4 );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        r00 = _mm256_add_pd( r00, _mm256_loadu_pd( solids[0] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[0] + (n  )*(n+1) + m, r00 );
        r10 = _mm256_add_pd( r10, _mm256_loadu_pd( solids[1] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[1] + (n  )*(n+1) + m, r10 );
        r20 = _mm256_add_pd( r20, _mm256_loadu_pd( solids[2] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[2] + (n  )*(n+1) + m, r20 );
        r30 = _mm256_add_pd( r30, _mm256_loadu_pd( solids[3] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[3] + (n  )*(n+1) + m, r30 );
        r01 = _mm256_add_pd( r01, _mm256_loadu_pd( solids[4] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[4] + (n  )*(n+1) + m, r01 );
        r11 = _mm256_add_pd( r11, _mm256_loadu_pd( solids[5] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[5] + (n  )*(n+1) + m, r11 );
        r21 = _mm256_add_pd( r21, _mm256_loadu_pd( solids[6] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[6] + (n  )*(n+1) + m, r21 );
        r31 = _mm256_add_pd( r31, _mm256_loadu_pd( solids[7] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[7] + (n  )*(n+1) + m, r31 );

        r00 = _mm256_load_pd( imag_in + (m+0)*16 + 0 );
        r01 = _mm256_load_pd( imag_in + (m+0)*16 + 4 );
        r10 = _mm256_load_pd( imag_in + (m+1)*16 + 0 );
        r11 = _mm256_load_pd( imag_in + (m+1)*16 + 4 );
        r20 = _mm256_load_pd( imag_in + (m+2)*16 + 0 );
        r21 = _mm256_load_pd( imag_in + (m+2)*16 + 4 );
        r30 = _mm256_load_pd( imag_in + (m+3)*16 + 0 );
        r31 = _mm256_load_pd( imag_in + (m+3)*16 + 4 );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        r00 = _mm256_add_pd( r00, _mm256_loadu_pd( solids[0] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[0] + (n+1)*(n+1) + m, r00 );
        r10 = _mm256_add_pd( r10, _mm256_loadu_pd( solids[1] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[1] + (n+1)*(n+1) + m, r10 );
        r20 = _mm256_add_pd( r20, _mm256_loadu_pd( solids[2] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[2] + (n+1)*(n+1) + m, r20 );
        r30 = _mm256_add_pd( r30, _mm256_loadu_pd( solids[3] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[3] + (n+1)*(n+1) + m, r30 );
        r01 = _mm256_add_pd( r01, _mm256_loadu_pd( solids[4] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[4] + (n+1)*(n+1) + m, r01 );
        r11 = _mm256_add_pd( r11, _mm256_loadu_pd( solids[5] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[5] + (n+1)*(n+1) + m, r11 );
        r21 = _mm256_add_pd( r21, _mm256_loadu_pd( solids[6] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[6] + (n+1)*(n+1) + m, r21 );
        r31 = _mm256_add_pd( r31, _mm256_loadu_pd( solids[7] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[7] + (n+1)*(n+1) + m, r31 );

        r00 = _mm256_load_pd( real_in + (m+0)*16 +  8 );
        r01 = _mm256_load_pd( real_in + (m+0)*16 + 12 );
        r10 = _mm256_load_pd( real_in + (m+1)*16 +  8 );
        r11 = _mm256_load_pd( real_in + (m+1)*16 + 12 );
        r20 = _mm256_load_pd( real_in + (m+2)*16 +  8 );
        r21 = _mm256_load_pd( real_in + (m+2)*16 + 12 );
        r30 = _mm256_load_pd( real_in + (m+3)*16 +  8 );
        r31 = _mm256_load_pd( real_in + (m+3)*16 + 12 );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        r00 = _mm256_add_pd( r00, _mm256_loadu_pd( solids[ 8] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[ 8] + (n  )*(n+1) + m, r00 );
        r10 = _mm256_add_pd( r10, _mm256_loadu_pd( solids[ 9] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[ 9] + (n  )*(n+1) + m, r10 );
        r20 = _mm256_add_pd( r20, _mm256_loadu_pd( solids[10] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[10] + (n  )*(n+1) + m, r20 );
        r30 = _mm256_add_pd( r30, _mm256_loadu_pd( solids[11] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[11] + (n  )*(n+1) + m, r30 );
        r01 = _mm256_add_pd( r01, _mm256_loadu_pd( solids[12] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[12] + (n  )*(n+1) + m, r01 );
        r11 = _mm256_add_pd( r11, _mm256_loadu_pd( solids[13] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[13] + (n  )*(n+1) + m, r11 );
        r21 = _mm256_add_pd( r21, _mm256_loadu_pd( solids[14] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[14] + (n  )*(n+1) + m, r21 );
        r31 = _mm256_add_pd( r31, _mm256_loadu_pd( solids[15] + (n  )*(n+1) + m ) );
        _mm256_storeu_pd( solids[15] + (n  )*(n+1) + m, r31 );

        r00 = _mm256_load_pd( imag_in + (m+0)*16 +  8 );
        r01 = _mm256_load_pd( imag_in + (m+0)*16 + 12 );
        r10 = _mm256_load_pd( imag_in + (m+1)*16 +  8 );
        r11 = _mm256_load_pd( imag_in + (m+1)*16 + 12 );
        r20 = _mm256_load_pd( imag_in + (m+2)*16 +  8 );
        r21 = _mm256_load_pd( imag_in + (m+2)*16 + 12 );
        r30 = _mm256_load_pd( imag_in + (m+3)*16 +  8 );
        r31 = _mm256_load_pd( imag_in + (m+3)*16 + 12 );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        r00 = _mm256_add_pd( r00, _mm256_loadu_pd( solids[ 8] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[ 8] + (n+1)*(n+1) + m, r00 );
        r10 = _mm256_add_pd( r10, _mm256_loadu_pd( solids[ 9] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[ 9] + (n+1)*(n+1) + m, r10 );
        r20 = _mm256_add_pd( r20, _mm256_loadu_pd( solids[10] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[10] + (n+1)*(n+1) + m, r20 );
        r30 = _mm256_add_pd( r30, _mm256_loadu_pd( solids[11] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[11] + (n+1)*(n+1) + m, r30 );
        r01 = _mm256_add_pd( r01, _mm256_loadu_pd( solids[12] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[12] + (n+1)*(n+1) + m, r01 );
        r11 = _mm256_add_pd( r11, _mm256_loadu_pd( solids[13] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[13] + (n+1)*(n+1) + m, r11 );
        r21 = _mm256_add_pd( r21, _mm256_loadu_pd( solids[14] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[14] + (n+1)*(n+1) + m, r21 );
        r31 = _mm256_add_pd( r31, _mm256_loadu_pd( solids[15] + (n+1)*(n+1) + m ) );
        _mm256_storeu_pd( solids[15] + (n+1)*(n+1) + m, r31 );
    }
    
    for ( size_t l  = 0; l  < 16; ++l  )
    for ( size_t mm = m; mm <= n; ++mm )
    {
        solids[l][ (n  )*(n+1) + mm ] += real_in[ mm*16 + l ];
        solids[l][ (n+1)*(n+1) + mm ] += imag_in[ mm*16 + l ];
    }
}

void microkernel_double_avx512f::
solid2buf( const double *const *p_solids, const double *const zeros, const size_t *P,
                 double *real_out, double *imag_out, size_t n ) const noexcept
{
    const double *const solids[ 16 ] =
    {
        ( n < P[ 0] ) ? p_solids[ 0] : zeros,
        ( n < P[ 1] ) ? p_solids[ 1] : zeros,
        ( n < P[ 2] ) ? p_solids[ 2] : zeros,
        ( n < P[ 3] ) ? p_solids[ 3] : zeros,
        ( n < P[ 4] ) ? p_solids[ 4] : zeros,
        ( n < P[ 5] ) ? p_solids[ 5] : zeros,
        ( n < P[ 6] ) ? p_solids[ 6] : zeros,
        ( n < P[ 7] ) ? p_solids[ 7] : zeros,
        ( n < P[ 8] ) ? p_solids[ 8] : zeros,
        ( n < P[ 9] ) ? p_solids[ 9] : zeros,
        ( n < P[10] ) ? p_solids[10] : zeros,
        ( n < P[11] ) ? p_solids[11] : zeros,
        ( n < P[12] ) ? p_solids[12] : zeros,
        ( n < P[13] ) ? p_solids[13] : zeros,
        ( n < P[14] ) ? p_solids[14] : zeros,
        ( n < P[15] ) ? p_solids[15] : zeros
    };

    size_t m;
    __m256d r00, r01, r10, r11, r20, r21, r30, r31;
    for ( m = 0; m + 3 <= n; m += 4 )
    {
        r00 = _mm256_loadu_pd( solids[0] + (n  )*(n+1) + m );
        r10 = _mm256_loadu_pd( solids[1] + (n  )*(n+1) + m );
        r20 = _mm256_loadu_pd( solids[2] + (n  )*(n+1) + m );
        r30 = _mm256_loadu_pd( solids[3] + (n  )*(n+1) + m );
        r01 = _mm256_loadu_pd( solids[4] + (n  )*(n+1) + m );
        r11 = _mm256_loadu_pd( solids[5] + (n  )*(n+1) + m );
        r21 = _mm256_loadu_pd( solids[6] + (n  )*(n+1) + m );
        r31 = _mm256_loadu_pd( solids[7] + (n  )*(n+1) + m );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        _mm256_store_pd( real_out + (m+0)*16 + 0, r00 );
        _mm256_store_pd( real_out + (m+0)*16 + 4, r01 );
        _mm256_store_pd( real_out + (m+1)*16 + 0, r10 );
        _mm256_store_pd( real_out + (m+1)*16 + 4, r11 );
        _mm256_store_pd( real_out + (m+2)*16 + 0, r20 );
        _mm256_store_pd( real_out + (m+2)*16 + 4, r21 );
        _mm256_store_pd( real_out + (m+3)*16 + 0, r30 );
        _mm256_store_pd( real_out + (m+3)*16 + 4, r31 );

        r00 = _mm256_loadu_pd( solids[0] + (n+1)*(n+1) + m );
        r10 = _mm256_loadu_pd( solids[1] + (n+1)*(n+1) + m );
        r20 = _mm256_loadu_pd( solids[2] + (n+1)*(n+1) + m );
        r30 = _mm256_loadu_pd( solids[3] + (n+1)*(n+1) + m );
        r01 = _mm256_loadu_pd( solids[4] + (n+1)*(n+1) + m );
        r11 = _mm256_loadu_pd( solids[5] + (n+1)*(n+1) + m );
        r21 = _mm256_loadu_pd( solids[6] + (n+1)*(n+1) + m );
        r31 = _mm256_loadu_pd( solids[7] + (n+1)*(n+1) + m );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        _mm256_store_pd( imag_out + (m+0)*16 + 0, r00 );
        _mm256_store_pd( imag_out + (m+0)*16 + 4, r01 );
        _mm256_store_pd( imag_out + (m+1)*16 + 0, r10 );
        _mm256_store_pd( imag_out + (m+1)*16 + 4, r11 );
        _mm256_store_pd( imag_out + (m+2)*16 + 0, r20 );
        _mm256_store_pd( imag_out + (m+2)*16 + 4, r21 );
        _mm256_store_pd( imag_out + (m+3)*16 + 0, r30 );
        _mm256_store_pd( imag_out + (m+3)*16 + 4, r31 );

        r00 = _mm256_loadu_pd( solids[ 8] + (n  )*(n+1) + m );
        r10 = _mm256_loadu_pd( solids[ 9] + (n  )*(n+1) + m );
        r20 = _mm256_loadu_pd( solids[10] + (n  )*(n+1) + m );
        r30 = _mm256_loadu_pd( solids[11] + (n  )*(n+1) + m );
        r01 = _mm256_loadu_pd( solids[12] + (n  )*(n+1) + m );
        r11 = _mm256_loadu_pd( solids[13] + (n  )*(n+1) + m );
        r21 = _mm256_loadu_pd( solids[14] + (n  )*(n+1) + m );
        r31 = _mm256_loadu_pd( solids[15] + (n  )*(n+1) + m );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        _mm256_store_pd( real_out + (m+0)*16 +  8, r00 );
        _mm256_store_pd( real_out + (m+0)*16 + 12, r01 );
        _mm256_store_pd( real_out + (m+1)*16 +  8, r10 );
        _mm256_store_pd( real_out + (m+1)*16 + 12, r11 );
        _mm256_store_pd( real_out + (m+2)*16 +  8, r20 );
        _mm256_store_pd( real_out + (m+2)*16 + 12, r21 );
        _mm256_store_pd( real_out + (m+3)*16 +  8, r30 );
        _mm256_store_pd( real_out + (m+3)*16 + 12, r31 );

        r00 = _mm256_loadu_pd( solids[ 8] + (n+1)*(n+1) + m );
        r10 = _mm256_loadu_pd( solids[ 9] + (n+1)*(n+1) + m );
        r20 = _mm256_loadu_pd( solids[10] + (n+1)*(n+1) + m );
        r30 = _mm256_loadu_pd( solids[11] + (n+1)*(n+1) + m );
        r01 = _mm256_loadu_pd( solids[12] + (n+1)*(n+1) + m );
        r11 = _mm256_loadu_pd( solids[13] + (n+1)*(n+1) + m );
        r21 = _mm256_loadu_pd( solids[14] + (n+1)*(n+1) + m );
        r31 = _mm256_loadu_pd( solids[15] + (n+1)*(n+1) + m );
        transpose(r00,r10,r20,r30);
        transpose(r01,r11,r21,r31);
        _mm256_store_pd( imag_out + (m+0)*16 +  8, r00 );
        _mm256_store_pd( imag_out + (m+0)*16 + 12, r01 );
        _mm256_store_pd( imag_out + (m+1)*16 +  8, r10 );
        _mm256_store_pd( imag_out + (m+1)*16 + 12, r11 );
        _mm256_store_pd( imag_out + (m+2)*16 +  8, r20 );
        _mm256_store_pd( imag_out + (m+2)*16 + 12, r20 );
        _mm256_store_pd( imag_out + (m+3)*16 +  8, r30 );
        _mm256_store_pd( imag_out + (m+3)*16 + 12, r31 );
    }

    for ( size_t l  = 0; l  < 16; ++l  )
    for ( size_t mm = m; mm <= n; ++mm )
    {
        real_out[ mm*16 + l ] = solids[l][ (n  )*(n+1) + mm ];
        imag_out[ mm*16 + l ] = solids[l][ (n+1)*(n+1) + mm ];
    }
}

#undef transpose

}

#ifdef __llvm__
#pragma clang attribute pop
#endif

#endif // Check for amd64

