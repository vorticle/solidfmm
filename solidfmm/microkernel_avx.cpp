/*
 * Copyright (C) 2021, 2022, 2023 Matthias Kirchhart
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
#include <solidfmm/microkernel_avx.hpp>

#ifdef __x86_64__

#ifdef __llvm__
#pragma clang attribute push(__attribute__((target("avx,fma"))), apply_to=any(function))
#elif __GNUG__
#pragma GCC target   ("avx,fma")
#pragma GCC optimize ("O2")
#endif

#include <immintrin.h>
#include <stdexcept>

namespace solidfmm
{


///////////////////////
// Single Precision. //
///////////////////////


bool microkernel_float_avx::available() noexcept
{
    int result_avx  = __builtin_cpu_supports( "avx" );
    int result_fma  = __builtin_cpu_supports( "fma" );

    return result_avx && result_fma;
}

microkernel_float_avx::microkernel_float_avx(): microkernel { 6, 16, 64 }
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_float_avx: "
                                   "AVX instructions unavailable on this CPU." };
}

void microkernel_float_avx::
euler( const float *x,   const float *y, const float *z,
       float *r,         float *rinv,
       float *cos_alpha, float *sin_alpha,
       float *cos_beta,  float *sin_beta,
       size_t k ) const noexcept
{
    __m256 ones  = _mm256_set1_ps(1.0f);
    __m256 zeros = _mm256_setzero_ps();
    __m256 r0, r1, rinv0, rinv1; 
    __m256 ca0, sa0, ca1, sa1;   // cos/sin alpha
    __m256 cb0, sb0, cb1, sb1;   // cos/sin beta

    if ( k-- )
    {
        _mm256_store_ps( r            , ones  );
        _mm256_store_ps( r         + 8, ones  );
        _mm256_store_ps( rinv         , ones  );
        _mm256_store_ps( rinv      + 8, ones  );
        _mm256_store_ps( cos_alpha    , ones  );
        _mm256_store_ps( cos_alpha + 8, ones  );
        _mm256_store_ps( sin_alpha    , zeros );
        _mm256_store_ps( sin_alpha + 8, zeros );
        _mm256_store_ps( cos_beta     , ones  );
        _mm256_store_ps( cos_beta  + 8, ones  );
        _mm256_store_ps( sin_beta     , zeros );
        _mm256_store_ps( sin_beta  + 8, zeros );

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
        __m256 x0, x1, y0, y1, z0, z1;
        __m256 rxyinv0, rxyinv1, rxy0, rxy1;

        x0   = _mm256_load_ps(x+0); 
        x1   = _mm256_load_ps(x+8); 
        y0   = _mm256_load_ps(y+0); 
        y1   = _mm256_load_ps(y+8); 
        z0   = _mm256_load_ps(z+0); 
        z1   = _mm256_load_ps(z+8); 

        rxy0 = _mm256_mul_ps(x0,x0);
        rxy1 = _mm256_mul_ps(x1,x1);
        rxy0 = _mm256_fmadd_ps(y0,y0,rxy0);
        rxy1 = _mm256_fmadd_ps(y1,y1,rxy1);
        r0   = _mm256_fmadd_ps(z0,z0,rxy0);
        r1   = _mm256_fmadd_ps(z1,z1,rxy1);

        rxy0 = _mm256_sqrt_ps( rxy0 );
        rxy1 = _mm256_sqrt_ps( rxy1 );
        r0   = _mm256_sqrt_ps( r0 );
        r1   = _mm256_sqrt_ps( r1 );
        
        rxyinv0 = _mm256_div_ps( ones, rxy0 );
        rxyinv1 = _mm256_div_ps( ones, rxy1 );
        rinv0   = _mm256_div_ps( ones, r0 );
        rinv1   = _mm256_div_ps( ones, r1 );

        ca0 = _mm256_mul_ps( y0, rxyinv0 );
        ca1 = _mm256_mul_ps( y1, rxyinv1 );
        sa0 = _mm256_mul_ps( x0, rxyinv0 );
        sa1 = _mm256_mul_ps( x1, rxyinv1 );

        __m256 iszero0 = _mm256_cmp_ps( rxy0, zeros, _CMP_EQ_UQ );
        __m256 iszero1 = _mm256_cmp_ps( rxy1, zeros, _CMP_EQ_UQ );
        ca0 = _mm256_andnot_ps( iszero0, ca0 );
        ca1 = _mm256_andnot_ps( iszero1, ca1 );
        ca0 = _mm256_add_ps( ca0, _mm256_and_ps(iszero0,ones) );
        ca1 = _mm256_add_ps( ca1, _mm256_and_ps(iszero1,ones) );
        sa0 = _mm256_andnot_ps( iszero0, sa0 );
        sa1 = _mm256_andnot_ps( iszero1, sa1 );

        cb0 = _mm256_mul_ps( z0, rinv0 );
        cb1 = _mm256_mul_ps( z1, rinv1 );
        sb0 = _mm256_fnmadd_ps( rxy0, rinv0, zeros );
        sb1 = _mm256_fnmadd_ps( rxy1, rinv1, zeros );

        _mm256_store_ps( r            , r0 );
        _mm256_store_ps( r         + 8, r1 );
        _mm256_store_ps( rinv         , rinv0 );
        _mm256_store_ps( rinv      + 8, rinv1 );
        _mm256_store_ps( cos_alpha    , ca0 );
        _mm256_store_ps( cos_alpha + 8, ca1 );
        _mm256_store_ps( sin_alpha    , sa0 );
        _mm256_store_ps( sin_alpha + 8, sa1 );
        _mm256_store_ps( cos_beta     , cb0 );
        _mm256_store_ps( cos_beta  + 8, cb1 );
        _mm256_store_ps( sin_beta     , sb0 );
        _mm256_store_ps( sin_beta  + 8, sb1 );
    }
    else return;

    while ( k-- )
    {
        __m256 rn0, rn1, rninv0, rninv1;
        __m256 cn0, sn0, cn1, sn1, res0, res1;
       
        // r^{n+1}    = r^{ n} * r; 
        // r^{-(n+1)} = r^{-n} * r^{-1}; 
        rn0    = _mm256_load_ps( r     ); 
        rn1    = _mm256_load_ps( r + 8 ); 
        rninv0 = _mm256_load_ps( rinv     ); 
        rninv1 = _mm256_load_ps( rinv + 8 ); 
        
        rn0    = _mm256_mul_ps( rn0, r0 );
        rn1    = _mm256_mul_ps( rn1, r1 );
        rninv0 = _mm256_mul_ps( rninv0, rinv0 );
        rninv1 = _mm256_mul_ps( rninv1, rinv1 );
        _mm256_store_ps( r    + 16, rn0 );
        _mm256_store_ps( r    + 24, rn1 );
        _mm256_store_ps( rinv + 16, rninv0 );
        _mm256_store_ps( rinv + 24, rninv1 );

        // cos(alpha*(n+1)) = cos(alpha*n)*cos(alpha) - sin(alpha*n)*sin(alpha);
        // sin(alpha*(n+1)) = sin(alpha*n)*cos(alpha) + cos(alpha*n)*sin(alpha);
        cn0  = _mm256_load_ps( cos_alpha + 0 );
        cn1  = _mm256_load_ps( cos_alpha + 8 );
        sn0  = _mm256_load_ps( sin_alpha + 0 );
        sn1  = _mm256_load_ps( sin_alpha + 8 );
        res0 = _mm256_mul_ps(cn0,ca0);
        res1 = _mm256_mul_ps(cn1,ca1);
        res0 = _mm256_fnmadd_ps(sn0,sa0,res0);
        res1 = _mm256_fnmadd_ps(sn1,sa1,res1);
        _mm256_store_ps( cos_alpha + 16, res0 );
        _mm256_store_ps( cos_alpha + 24, res1 );
        res0 = _mm256_mul_ps(sn0,ca0);
        res1 = _mm256_mul_ps(sn1,ca1);
        res0 = _mm256_fmadd_ps(cn0,sa0,res0);
        res1 = _mm256_fmadd_ps(cn1,sa1,res1);
        _mm256_store_ps( sin_alpha + 16, res0 );
        _mm256_store_ps( sin_alpha + 24, res1 );

        // cos(beta*(n+1)) = cos(beta*n)*cos(beta) - sin(beta*n)*sin(beta);
        // sin(beta*(n+1)) = sin(beta*n)*cos(beta) + cos(beta*n)*sin(beta);
        cn0  = _mm256_load_ps( cos_beta + 0 );
        cn1  = _mm256_load_ps( cos_beta + 8 );
        sn0  = _mm256_load_ps( sin_beta + 0 );
        sn1  = _mm256_load_ps( sin_beta + 8 );
        res0 = _mm256_mul_ps(cn0,cb0);
        res1 = _mm256_mul_ps(cn1,cb1);
        res0 = _mm256_fnmadd_ps(sn0,sb0,res0);
        res1 = _mm256_fnmadd_ps(sn1,sb1,res1);
        _mm256_store_ps( cos_beta + 16, res0 );
        _mm256_store_ps( cos_beta + 24, res1 );
        res0 = _mm256_mul_ps(sn0,cb0);
        res1 = _mm256_mul_ps(sn1,cb1);
        res0 = _mm256_fmadd_ps(cn0,sb0,res0);
        res1 = _mm256_fmadd_ps(cn1,sb1,res1);
        _mm256_store_ps( sin_beta + 16, res0 );
        _mm256_store_ps( sin_beta + 24, res1 );
        
        r         = r         + 16;
        rinv      = rinv      + 16;
        cos_alpha = cos_alpha + 16;
        sin_alpha = sin_alpha + 16;
        cos_beta  = cos_beta  + 16;
        sin_beta  = sin_beta  + 16;
    }
}

void microkernel_float_avx::
rotscale( const float *cos,      const float *sin,      const float *scale,
          const float *real_in,  const float *imag_in,
                float *__restrict__ real_out,      
                float *__restrict__ imag_out,
          size_t k, bool forward ) const noexcept
{
    __m256 scale0, scale1;
    __m256 cos0, cos1, sin0, sin1;
    __m256 rin0, rin1;
    __m256 iin0, iin1;
    __m256 tmp0, tmp1;

    scale0 = _mm256_load_ps( scale     );
    scale1 = _mm256_load_ps( scale + 8 );
    if ( forward )
    {
        while ( k-- )
        {
            cos0 = _mm256_load_ps( cos     );
            cos1 = _mm256_load_ps( cos + 8 );
            sin0 = _mm256_load_ps( sin     );
            sin1 = _mm256_load_ps( sin + 8 );
            rin0 = _mm256_load_ps( real_in     );
            rin1 = _mm256_load_ps( real_in + 8 );
            iin0 = _mm256_load_ps( imag_in     );
            iin1 = _mm256_load_ps( imag_in + 8 );

            tmp0 = _mm256_mul_ps( cos0, rin0 );
            tmp1 = _mm256_mul_ps( cos1, rin1 );
            tmp0 = _mm256_fnmadd_ps( sin0, iin0, tmp0 );
            tmp1 = _mm256_fnmadd_ps( sin1, iin1, tmp1 );
            tmp0 = _mm256_mul_ps( scale0, tmp0 );
            tmp1 = _mm256_mul_ps( scale1, tmp1 );
            _mm256_store_ps( real_out    , tmp0 );
            _mm256_store_ps( real_out + 8, tmp1 );

            tmp0 = _mm256_mul_ps( sin0, rin0 );
            tmp1 = _mm256_mul_ps( sin1, rin1 );
            tmp0 = _mm256_fmadd_ps( cos0, iin0, tmp0 );
            tmp1 = _mm256_fmadd_ps( cos1, iin1, tmp1 );
            tmp0 = _mm256_mul_ps( scale0, tmp0 );
            tmp1 = _mm256_mul_ps( scale1, tmp1 );
            _mm256_store_ps( imag_out    , tmp0 );
            _mm256_store_ps( imag_out + 8, tmp1 );

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
            cos0 = _mm256_load_ps( cos     );
            cos1 = _mm256_load_ps( cos + 8 );
            sin0 = _mm256_load_ps( sin     );
            sin1 = _mm256_load_ps( sin + 8 );
            rin0 = _mm256_load_ps( real_in     );
            rin1 = _mm256_load_ps( real_in + 8 );
            iin0 = _mm256_load_ps( imag_in     );
            iin1 = _mm256_load_ps( imag_in + 8 );

            tmp0 = _mm256_mul_ps( cos0, rin0 );
            tmp1 = _mm256_mul_ps( cos1, rin1 );
            tmp0 = _mm256_fmadd_ps( sin0, iin0, tmp0 );
            tmp1 = _mm256_fmadd_ps( sin1, iin1, tmp1 );
            tmp0 = _mm256_mul_ps( scale0, tmp0 );
            tmp1 = _mm256_mul_ps( scale1, tmp1 );
            _mm256_store_ps( real_out    , tmp0 );
            _mm256_store_ps( real_out + 8, tmp1 );

            tmp0 = _mm256_mul_ps( sin0, rin0 );
            tmp1 = _mm256_mul_ps( sin1, rin1 );
            tmp0 = _mm256_fmsub_ps( cos0, iin0, tmp0 );
            tmp1 = _mm256_fmsub_ps( cos1, iin1, tmp1 );
            tmp0 = _mm256_mul_ps( scale0, tmp0 );
            tmp1 = _mm256_mul_ps( scale1, tmp1 );
            _mm256_store_ps( imag_out    , tmp0 );
            _mm256_store_ps( imag_out + 8, tmp1 );

            cos      = cos      + 16;
            sin      = sin      + 16;
            real_in  = real_in  + 16;
            real_out = real_out + 16;
            imag_in  = imag_in  + 16;
            imag_out = imag_out + 16;
        }
    }
}

void microkernel_float_avx::
swap( const float *mat, const float *in,
            float *out, size_t k, bool pattern ) const noexcept
{
    bool k_odd = k &  1;
         k     = k >> 1;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorps   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorps   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorps   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorps   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorps   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorps   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorps   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorps   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorps   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorps   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorps   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        "testq    %[k], %[k]                     \n\t"
        "jz       .Lcheckodd%=                   \n\t"
        "                                        \n\t"
        ".align 16                               \n\t"
        ".Lloop%=:                               \n\t"
        "                                        \n\t"
        "vmovaps    (%[in]), %%ymm12             \n\t" // Even iteration.
        "vmovaps  32(%[in]), %%ymm13             \n\t"
        "                                        \n\t"
        "vbroadcastss  (%[mat]),%%ymm14          \n\t"
        "vbroadcastss 4(%[mat]),%%ymm15          \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm0     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm1     \n\t"
        "vfmadd231ps  %%ymm12,%%ymm15,%%ymm4     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm15,%%ymm5     \n\t"
        "                                        \n\t"
        "vbroadcastss 8(%[mat]),%%ymm14          \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm8     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm9     \n\t"
        "                                        \n\t"
        "vmovaps   64(%[in]), %%ymm12            \n\t" // Odd iteration.
        "vmovaps   96(%[in]), %%ymm13            \n\t"
        "                                        \n\t"
        "vbroadcastss 12(%[mat]),%%ymm14         \n\t"
        "vbroadcastss 16(%[mat]),%%ymm15         \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm2     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm3     \n\t"
        "vfmadd231ps  %%ymm12,%%ymm15,%%ymm6     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm15,%%ymm7     \n\t"
        "                                        \n\t"
        "vbroadcastss 20(%[mat]),%%ymm14         \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm10    \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm11    \n\t"
        "                                        \n\t"
        "addq    $128,%[in]                      \n\t"
        "addq     $24,%[mat]                     \n\t"
        "decq         %[k]                       \n\t"
        "jnz          .Lloop%=                   \n\t"
        "                                        \n\t"
        ".Lcheckodd%=:                           \n\t" // Remaining even iteration, if applicable.
        "testb        %[k_odd],%[k_odd]          \n\t"
        "jz           .Lstoreresult%=            \n\t"
        "vmovaps    (%[in]), %%ymm12             \n\t" // Even iteration.
        "vmovaps  32(%[in]), %%ymm13             \n\t"
        "                                        \n\t"
        "vbroadcastss  (%[mat]),%%ymm14          \n\t"
        "vbroadcastss 4(%[mat]),%%ymm15          \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm0     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm1     \n\t"
        "vfmadd231ps  %%ymm12,%%ymm15,%%ymm4     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm15,%%ymm5     \n\t"
        "                                        \n\t"
        "vbroadcastss 8(%[mat]),%%ymm14          \n\t"
        "vfmadd231ps  %%ymm12,%%ymm14,%%ymm8     \n\t"
        "vfmadd231ps  %%ymm13,%%ymm14,%%ymm9     \n\t"
        "                                        \n\t"
        "                                        \n\t"
        ".Lstoreresult%=:                        \n\t"
        "testb %[pattern],%[pattern]             \n\t"
        "jz    .Lnegativepattern%=               \n\t"
        "vmovaps     %%ymm0,    (%[out])         \n\t"
        "vmovaps     %%ymm1,  32(%[out])         \n\t"
        "vmovaps     %%ymm2,  64(%[out])         \n\t"
        "vmovaps     %%ymm3,  96(%[out])         \n\t"
        "vmovaps     %%ymm4, 128(%[out])         \n\t"
        "vmovaps     %%ymm5, 160(%[out])         \n\t"
        "vmovaps     %%ymm6, 192(%[out])         \n\t"
        "vmovaps     %%ymm7, 224(%[out])         \n\t"
        "vmovaps     %%ymm8, 256(%[out])         \n\t"
        "vmovaps     %%ymm9, 288(%[out])         \n\t"
        "vmovaps     %%ymm10,320(%[out])         \n\t"
        "vmovaps     %%ymm11,352(%[out])         \n\t"
        "jmp .Lfinish%=                          \n\t"
        "                                        \n\t"
        ".Lnegativepattern%=:                    \n\t"
        "vmovaps     %%ymm2,    (%[out])         \n\t"
        "vmovaps     %%ymm3,  32(%[out])         \n\t"
        "vmovaps     %%ymm0,  64(%[out])         \n\t"
        "vmovaps     %%ymm1,  96(%[out])         \n\t"
        "vmovaps     %%ymm6, 128(%[out])         \n\t"
        "vmovaps     %%ymm7, 160(%[out])         \n\t"
        "vmovaps     %%ymm4, 192(%[out])         \n\t"
        "vmovaps     %%ymm5, 224(%[out])         \n\t"
        "vmovaps     %%ymm10,256(%[out])         \n\t"
        "vmovaps     %%ymm11,288(%[out])         \n\t"
        "vmovaps     %%ymm8, 320(%[out])         \n\t"
        "vmovaps     %%ymm9, 352(%[out])         \n\t"
        "                                        \n\t"
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
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}

void microkernel_float_avx::
zm2l( const float* fac, const float* in,
            float* out, size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;
    const float signs { -0.0f };

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorps   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorps   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorps   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorps   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorps   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorps   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorps   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorps   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorps   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorps   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorps   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        ".align 16                               \n\t"
        ".Lloop%=:                               \n\t" // do 
        "                                        \n\t" // {
        "vmovaps          (%[in]), %%ymm12       \n\t" // b0   = _mm256_load_pd(in+0);
        "vmovaps        32(%[in]), %%ymm13       \n\t" // b1   = _mm256_load_pd(in+8);
        "                                        \n\t"
        "vbroadcastss    (%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+0); 
        "vbroadcastss   4(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+1);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm0  \n\t" // c00  = _mm256_fmadd_ps(b0,tmp0,c00)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm1  \n\t" // c01  = _mm256_fmadd_ps(b1,tmp0,c01)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm2  \n\t" // c10  = _mm256_fmadd_ps(b0,tmp1,c10)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm3  \n\t" // c11  = _mm256_fmadd_ps(b1,tmp1,c11)
        "                                        \n\t"
        "vbroadcastss   8(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+2); 
        "vbroadcastss  12(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+3);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm4  \n\t" // c20  = _mm256_fmadd_ps(b0,tmp0,c20)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm5  \n\t" // c21  = _mm256_fmadd_ps(b1,tmp0,c21)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm6  \n\t" // c30  = _mm256_fmadd_ps(b0,tmp1,c30)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm7  \n\t" // c31  = _mm256_fmadd_ps(b1,tmp1,c31)
        "                                        \n\t"
        "vbroadcastss  16(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+4); 
        "vbroadcastss  20(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+5);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm8  \n\t" // c40  = _mm256_fmadd_ps(b0,tmp0,c40)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm9  \n\t" // c41  = _mm256_fmadd_ps(b1,tmp0,c41)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm10 \n\t" // c50  = _mm256_fmadd_ps(b0,tmp1,c50)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm11 \n\t" // c51  = _mm256_fmadd_ps(b1,tmp1,c51)
        "                                        \n\t"
        "addq          $64,%[in]                 \n\t" // in   = in + 16;
        "addq          $4, %[fac]                \n\t" // fac = fac + 1;
        "decq          %[k]                      \n\t" // k--;
        "jnz           .Lloop%=                  \n\t" // } while (k);
        "                                        \n\t"
        "vbroadcastss  %[signs], %%ymm15         \n\t" // tmp1 = _mm256_set1_ps( -0.0f );
        "testb         %[pattern], %[pattern]    \n\t" // if ( pattern )
        "jz           .Lnegativepattern%=        \n\t" // {
        "vxorps        %%ymm15, %%ymm0, %%ymm0   \n\t" // c00 = _mm256_xor_ps( c00, tmp1 )
        "vxorps        %%ymm15, %%ymm1, %%ymm1   \n\t" // c01 = _mm256_xor_ps( c01, tmp1 )
        "vxorps        %%ymm15, %%ymm4, %%ymm4   \n\t" // c20 = _mm256_xor_ps( c20, tmp1 )
        "vxorps        %%ymm15, %%ymm5, %%ymm5   \n\t" // c21 = _mm256_xor_ps( c21, tmp1 )
        "vxorps        %%ymm15, %%ymm8, %%ymm8   \n\t" // c40 = _mm256_xor_ps( c40, tmp1 )
        "vxorps        %%ymm15, %%ymm9, %%ymm9   \n\t" // c41 = _mm256_xor_ps( c41, tmp1 )
        "jmp           .Lstoreresult%=           \n\t" // } 
        "                                        \n\t" // else
        ".Lnegativepattern%=:                    \n\t" // {
        "vxorps        %%ymm15, %%ymm2 , %%ymm2  \n\t" // c10 = _mm256_xor_ps( c10, tmp1 )
        "vxorps        %%ymm15, %%ymm3 , %%ymm3  \n\t" // c11 = _mm256_xor_ps( c11, tmp1 )
        "vxorps        %%ymm15, %%ymm6 , %%ymm6  \n\t" // c30 = _mm256_xor_ps( c30, tmp1 )
        "vxorps        %%ymm15, %%ymm7 , %%ymm7  \n\t" // c31 = _mm256_xor_ps( c31, tmp1 )
        "vxorps        %%ymm15, %%ymm10, %%ymm10 \n\t" // c50 = _mm256_xor_ps( c50, tmp1 )
        "vxorps        %%ymm15, %%ymm11, %%ymm11 \n\t" // c51 = _mm256_xor_ps( c51, tmp1 )
        "                                        \n\t" // }
        ".Lstoreresult%=:                        \n\t"
        "vmovaps       %%ymm0,     (%[out])      \n\t" // _mm256_store_ps( out   , c00 );
        "vmovaps       %%ymm1,   32(%[out])      \n\t" // _mm256_store_ps( out+ 8, c01 );
        "vmovaps       %%ymm2,   64(%[out])      \n\t" // _mm256_store_ps( out+16, c10 );
        "vmovaps       %%ymm3,   96(%[out])      \n\t" // _mm256_store_ps( out+24, c11 );
        "vmovaps       %%ymm4,  128(%[out])      \n\t" // _mm256_store_ps( out+32, c20 );
        "vmovaps       %%ymm5,  160(%[out])      \n\t" // _mm256_store_ps( out+40, c21 );
        "vmovaps       %%ymm6,  192(%[out])      \n\t" // _mm256_store_ps( out+48, c30 );
        "vmovaps       %%ymm7,  224(%[out])      \n\t" // _mm256_store_ps( out+56, c31 );
        "vmovaps       %%ymm8,  256(%[out])      \n\t" // _mm256_store_ps( out+64, c40 );
        "vmovaps       %%ymm9,  288(%[out])      \n\t" // _mm256_store_ps( out+72, c41 );
        "vmovaps       %%ymm10, 320(%[out])      \n\t" // _mm256_store_ps( out+80, c50 );
        "vmovaps       %%ymm11, 352(%[out])      \n\t" // _mm256_store_ps( out+88, c51 );
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [signs]   "m"(signs)

        : // Clobbered registers
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}

void microkernel_float_avx::
zm2m( const float* fac, const float* in,
            float* out, size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorps   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorps   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorps   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorps   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorps   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorps   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorps   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorps   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorps   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorps   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorps   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorps   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" // do
        "                                        \n\t" // {
        "vmovaps          (%[in]), %%ymm12       \n\t" // b0   = _mm256_load_ps(in+0);
        "vmovaps        32(%[in]), %%ymm13       \n\t" // b1   = _mm256_load_ps(in+8);
        "                                        \n\t"
        "vbroadcastss    (%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+0); 
        "vbroadcastss   4(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+1);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm0  \n\t" // c50  = _mm256_fmadd_ps(b0,tmp0,c50)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm1  \n\t" // c51  = _mm256_fmadd_ps(b1,tmp0,c51)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm2  \n\t" // c40  = _mm256_fmadd_ps(b0,tmp1,c40)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm3  \n\t" // c41  = _mm256_fmadd_ps(b1,tmp1,c41)
        "                                        \n\t"
        "vbroadcastss   8(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+2); 
        "vbroadcastss  12(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+3);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm4  \n\t" // c30  = _mm256_fmadd_ps(b0,tmp0,c30)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm5  \n\t" // c31  = _mm256_fmadd_ps(b1,tmp0,c31)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm6  \n\t" // c20  = _mm256_fmadd_ps(b0,tmp1,c20)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm7  \n\t" // c21  = _mm256_fmadd_ps(b1,tmp1,c21)
        "                                        \n\t"
        "vbroadcastss  16(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_ss(fac+4); 
        "vbroadcastss  20(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_ss(fac+5);
        "vfmadd231ps   %%ymm12, %%ymm14, %%ymm8  \n\t" // c10  = _mm256_fmadd_ps(b0,tmp0,c10)
        "vfmadd231ps   %%ymm13, %%ymm14, %%ymm9  \n\t" // c11  = _mm256_fmadd_ps(b1,tmp0,c11)
        "vfmadd231ps   %%ymm12, %%ymm15, %%ymm10 \n\t" // c00  = _mm256_fmadd_ps(b0,tmp1,c00)
        "vfmadd231ps   %%ymm13, %%ymm15, %%ymm11 \n\t" // c01  = _mm256_fmadd_ps(b1,tmp1,c01)
        "                                        \n\t"
        "addq          $64,%[in]                 \n\t" // in  = in + 16;
        "addq          $4, %[fac]                \n\t" // fac = fac + 1;
        "decq          %[k]                      \n\t" // --k;
        "jnz           .Lloop%=                  \n\t" // } while (k);
        "                                        \n\t"
        "vmovaps       %%ymm10,    (%[out])      \n\t" // _mm256_store_ps( out   , c00 );
        "vmovaps       %%ymm11,  32(%[out])      \n\t" // _mm256_store_ps( out+ 8, c01 );
        "vmovaps       %%ymm8,   64(%[out])      \n\t" // _mm256_store_ps( out+16, c10 );
        "vmovaps       %%ymm9,   96(%[out])      \n\t" // _mm256_store_ps( out+24, c11 );
        "vmovaps       %%ymm6,  128(%[out])      \n\t" // _mm256_store_ps( out+32, c20 );
        "vmovaps       %%ymm7,  160(%[out])      \n\t" // _mm256_store_ps( out+40, c21 );
        "vmovaps       %%ymm4,  192(%[out])      \n\t" // _mm256_store_ps( out+48, c30 );
        "vmovaps       %%ymm5,  224(%[out])      \n\t" // _mm256_store_ps( out+56, c31 );
        "vmovaps       %%ymm2,  256(%[out])      \n\t" // _mm256_store_ps( out+64, c40 );
        "vmovaps       %%ymm3,  288(%[out])      \n\t" // _mm256_store_ps( out+72, c41 );
        "vmovaps       %%ymm0,  320(%[out])      \n\t" // _mm256_store_ps( out+80, c50 );
        "vmovaps       %%ymm1,  352(%[out])      \n\t" // _mm256_store_ps( out+88, c51 );
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out)

        : // Clobbered registers
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}


void microkernel_float_avx::
swap2trans_buf( const float *__restrict__ real_in, 
                const float *__restrict__ imag_in,
                      float **real_out, 
                      float **imag_out,
                      size_t n ) const noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        float *__restrict__ rdst = real_out[m] + (n-m)*16;
        float *__restrict__ idst = imag_out[m] + (n-m)*16;

        __m256 rtmp0 = _mm256_load_ps( real_in     );
        __m256 rtmp1 = _mm256_load_ps( real_in + 8 );
        __m256 itmp0 = _mm256_load_ps( imag_in     );
        __m256 itmp1 = _mm256_load_ps( imag_in + 8 );
        _mm256_store_ps( rdst    , rtmp0 );
        _mm256_store_ps( rdst + 8, rtmp1 );
        _mm256_store_ps( idst    , itmp0 );
        _mm256_store_ps( idst + 8, itmp1 );
        real_in += 16;
        imag_in += 16;
    }
}

void microkernel_float_avx::
trans2swap_buf( const float *const *const real_in, 
                const float *const *const imag_in,
                      float *__restrict__ real_out, 
                      float *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    using std::min;
    __m256 zeros = _mm256_setzero_ps();

    for ( size_t m = 0; m <= min(n,Pmax); ++m )
    {
        const float *__restrict__ rsrc = real_in[m] + (n-m)*16;
        const float *__restrict__ isrc = imag_in[m] + (n-m)*16;
        __m256 rtmp0 = _mm256_load_ps( rsrc     );
        __m256 rtmp1 = _mm256_load_ps( rsrc + 8 );
        __m256 itmp0 = _mm256_load_ps( isrc     );
        __m256 itmp1 = _mm256_load_ps( isrc + 8 );
        _mm256_store_ps( real_out    , rtmp0 );
        _mm256_store_ps( real_out + 8, rtmp1 );
        _mm256_store_ps( imag_out    , itmp0 );
        _mm256_store_ps( imag_out + 8, itmp1 );
        real_out += 16;
        imag_out += 16;
    }

    for ( size_t m = min(n,Pmax); m <= n; ++m )
    {
        _mm256_store_ps( real_out    , zeros );
        _mm256_store_ps( real_out + 8, zeros );
        _mm256_store_ps( imag_out    , zeros );
        _mm256_store_ps( imag_out + 8, zeros );
        real_out += 16;
        imag_out += 16;
    }
}

void microkernel_float_avx::
buf2solid( const float   *real_in, const float *imag_in, 
                 float  **p_solids,  float *trash, const size_t *P, size_t n ) const noexcept
{
    float *solids[ 16 ] =
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

    for ( size_t l = 0; l < 16; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        solids[l][ (n  )*(n+1) + m ] += real_in[ m*16 + l ];
        solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*16 + l ];
    } 
}

void microkernel_float_avx::
solid2buf( const float *const *p_solids, const float *const zeros, const size_t *P,
                 float *real_out, float *imag_out, size_t n ) const noexcept
{
    const float *const solids[ 16 ] =
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

    for ( size_t l = 0; l < 16; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        real_out[ m*16 + l ] = solids[l][ (n  )*(n+1) + m ];
        imag_out[ m*16 + l ] = solids[l][ (n+1)*(n+1) + m ];
    }
}

///////////////////////
// Double Precision. //
///////////////////////

bool microkernel_double_avx::available() noexcept
{
    int result_avx  = __builtin_cpu_supports( "avx" );
    int result_fma  = __builtin_cpu_supports( "fma" );

    return result_avx && result_fma;
}

microkernel_double_avx::microkernel_double_avx(): microkernel { 6, 8, 64 }
{
    if ( ! available() )
        throw std::runtime_error { "solidfmm::microkernel_double_avx: "
                                   "AVX instructions unavailable on this CPU." };
}

void microkernel_double_avx::
euler( const double *x,   const double *y, const double *z,
       double *r,         double *rinv,
       double *cos_alpha, double *sin_alpha,
       double *cos_beta,  double *sin_beta,
       size_t k ) const noexcept
{
    __m256d ones  = _mm256_set1_pd(1.0);
    __m256d zeros = _mm256_setzero_pd();
    __m256d r0, r1, rinv0, rinv1; 
    __m256d ca0, sa0, ca1, sa1;   // cos/sin alpha
    __m256d cb0, sb0, cb1, sb1;   // cos/sin beta

    if ( k-- )
    {
        _mm256_store_pd( r            , ones  );
        _mm256_store_pd( r         + 4, ones  );
        _mm256_store_pd( rinv         , ones  );
        _mm256_store_pd( rinv      + 4, ones  );
        _mm256_store_pd( cos_alpha    , ones  );
        _mm256_store_pd( cos_alpha + 4, ones  );
        _mm256_store_pd( sin_alpha    , zeros );
        _mm256_store_pd( sin_alpha + 4, zeros );
        _mm256_store_pd( cos_beta     , ones  );
        _mm256_store_pd( cos_beta  + 4, ones  );
        _mm256_store_pd( sin_beta     , zeros );
        _mm256_store_pd( sin_beta  + 4, zeros );

        r         = r         + 8;
        rinv      = rinv      + 8;
        cos_alpha = cos_alpha + 8;
        sin_alpha = sin_alpha + 8;
        cos_beta  = cos_beta  + 8;
        sin_beta  = sin_beta  + 8;
    }
    else return;

    if ( k-- )
    {
        __m256d x0, x1, y0, y1, z0, z1;
        __m256d rxyinv0, rxyinv1, rxy0, rxy1;

        x0   = _mm256_load_pd(x+0); 
        x1   = _mm256_load_pd(x+4); 
        y0   = _mm256_load_pd(y+0); 
        y1   = _mm256_load_pd(y+4); 
        z0   = _mm256_load_pd(z+0); 
        z1   = _mm256_load_pd(z+4); 

        rxy0 = _mm256_mul_pd(x0,x0);
        rxy1 = _mm256_mul_pd(x1,x1);
        rxy0 = _mm256_fmadd_pd(y0,y0,rxy0);
        rxy1 = _mm256_fmadd_pd(y1,y1,rxy1);
        r0   = _mm256_fmadd_pd(z0,z0,rxy0);
        r1   = _mm256_fmadd_pd(z1,z1,rxy1);

        rxy0 = _mm256_sqrt_pd( rxy0 );
        rxy1 = _mm256_sqrt_pd( rxy1 );
        r0   = _mm256_sqrt_pd( r0 );
        r1   = _mm256_sqrt_pd( r1 );

        rxyinv0 = _mm256_div_pd( ones, rxy0 );
        rxyinv1 = _mm256_div_pd( ones, rxy1 );
        rinv0   = _mm256_div_pd( ones, r0 );
        rinv1   = _mm256_div_pd( ones, r1 );

        ca0 = _mm256_mul_pd( y0, rxyinv0 );
        ca1 = _mm256_mul_pd( y1, rxyinv1 );
        sa0 = _mm256_mul_pd( x0, rxyinv0 );
        sa1 = _mm256_mul_pd( x1, rxyinv1 );

        __m256d iszero0 = _mm256_cmp_pd( rxy0, zeros, _CMP_EQ_UQ );
        __m256d iszero1 = _mm256_cmp_pd( rxy1, zeros, _CMP_EQ_UQ );
        ca0 = _mm256_andnot_pd( iszero0, ca0 );
        ca1 = _mm256_andnot_pd( iszero1, ca1 );
        ca0 = _mm256_add_pd(ca0,_mm256_and_pd(iszero0,ones) );
        ca1 = _mm256_add_pd(ca1,_mm256_and_pd(iszero1,ones) );
        sa0 = _mm256_andnot_pd( iszero0, sa0 );
        sa1 = _mm256_andnot_pd( iszero1, sa1 );

        cb0 = _mm256_mul_pd( z0, rinv0 );
        cb1 = _mm256_mul_pd( z1, rinv1 );
        sb0 = _mm256_fnmadd_pd( rxy0, rinv0, zeros );
        sb1 = _mm256_fnmadd_pd( rxy1, rinv1, zeros );

        _mm256_store_pd( r            , r0 );
        _mm256_store_pd( r         + 4, r1 );
        _mm256_store_pd( rinv         , rinv0 );
        _mm256_store_pd( rinv      + 4, rinv1 );
        _mm256_store_pd( cos_alpha    , ca0 );
        _mm256_store_pd( cos_alpha + 4, ca1 );
        _mm256_store_pd( sin_alpha    , sa0 );
        _mm256_store_pd( sin_alpha + 4, sa1 );
        _mm256_store_pd( cos_beta     , cb0 );
        _mm256_store_pd( cos_beta  + 4, cb1 );
        _mm256_store_pd( sin_beta     , sb0 );
        _mm256_store_pd( sin_beta  + 4, sb1 );
    }
    else return;

    while ( k-- )
    {
        __m256d rn0, rn1, rninv0, rninv1;
        __m256d cn0, sn0, cn1, sn1, res0, res1;
       
        // r^{n+1}    = r^{ n} * r; 
        // r^{-(n+1)} = r^{-n} * r^{-1}; 
        rn0    = _mm256_load_pd( r     ); 
        rn1    = _mm256_load_pd( r + 4 ); 
        rninv0 = _mm256_load_pd( rinv     ); 
        rninv1 = _mm256_load_pd( rinv + 4 ); 
        
        rn0    = _mm256_mul_pd( rn0, r0 );
        rn1    = _mm256_mul_pd( rn1, r1 );
        rninv0 = _mm256_mul_pd( rninv0, rinv0 );
        rninv1 = _mm256_mul_pd( rninv1, rinv1 );
        _mm256_store_pd( r    +  8, rn0 );
        _mm256_store_pd( r    + 12, rn1 );
        _mm256_store_pd( rinv +  8, rninv0 );
        _mm256_store_pd( rinv + 12, rninv1 );

        // cos(alpha*(n+1)) = cos(alpha*n)*cos(alpha) - sin(alpha*n)*sin(alpha);
        // sin(alpha*(n+1)) = sin(alpha*n)*cos(alpha) + cos(alpha*n)*sin(alpha);
        cn0  = _mm256_load_pd( cos_alpha     );
        cn1  = _mm256_load_pd( cos_alpha + 4 );
        sn0  = _mm256_load_pd( sin_alpha     );
        sn1  = _mm256_load_pd( sin_alpha + 4 );
        res0 = _mm256_mul_pd(cn0,ca0);
        res1 = _mm256_mul_pd(cn1,ca1);
        res0 = _mm256_fnmadd_pd(sn0,sa0,res0);
        res1 = _mm256_fnmadd_pd(sn1,sa1,res1);
        _mm256_store_pd( cos_alpha +  8, res0 );
        _mm256_store_pd( cos_alpha + 12, res1 );
        res0 = _mm256_mul_pd(sn0,ca0);
        res1 = _mm256_mul_pd(sn1,ca1);
        res0 = _mm256_fmadd_pd(cn0,sa0,res0);
        res1 = _mm256_fmadd_pd(cn1,sa1,res1);
        _mm256_store_pd( sin_alpha +  8, res0 );
        _mm256_store_pd( sin_alpha + 12, res1 );

        // cos(beta*(n+1)) = cos(beta*n)*cos(beta) - sin(beta*n)*sin(beta);
        // sin(beta*(n+1)) = sin(beta*n)*cos(beta) + cos(beta*n)*sin(beta);
        cn0  = _mm256_load_pd( cos_beta     );
        cn1  = _mm256_load_pd( cos_beta + 4 );
        sn0  = _mm256_load_pd( sin_beta     );
        sn1  = _mm256_load_pd( sin_beta + 4 );
        res0 = _mm256_mul_pd(cn0,cb0);
        res1 = _mm256_mul_pd(cn1,cb1);
        res0 = _mm256_fnmadd_pd(sn0,sb0,res0);
        res1 = _mm256_fnmadd_pd(sn1,sb1,res1);
        _mm256_store_pd( cos_beta +  8, res0 );
        _mm256_store_pd( cos_beta + 12, res1 );
        res0 = _mm256_mul_pd(sn0,cb0);
        res1 = _mm256_mul_pd(sn1,cb1);
        res0 = _mm256_fmadd_pd(cn0,sb0,res0);
        res1 = _mm256_fmadd_pd(cn1,sb1,res1);
        _mm256_store_pd( sin_beta +  8, res0 );
        _mm256_store_pd( sin_beta + 12, res1 );
        
        r         = r         + 8;
        rinv      = rinv      + 8;
        cos_alpha = cos_alpha + 8;
        sin_alpha = sin_alpha + 8;
        cos_beta  = cos_beta  + 8;
        sin_beta  = sin_beta  + 8;
    }
}

void microkernel_double_avx::
rotscale( const double *cos,      const double *sin,      const double *scale,
          const double *real_in,  const double *imag_in,
                double *__restrict__ real_out,      
                double *__restrict__ imag_out,
          size_t k, bool forward ) const noexcept
{
    __m256d scale0, scale1;
    __m256d cos0, cos1, sin0, sin1;
    __m256d rin0, rin1;
    __m256d iin0, iin1;
    __m256d tmp0, tmp1;

    scale0 = _mm256_load_pd( scale     );
    scale1 = _mm256_load_pd( scale + 4 );

    if ( forward )
    {
        while ( k-- )
        {
            cos0 = _mm256_load_pd( cos     );
            cos1 = _mm256_load_pd( cos + 4 );
            sin0 = _mm256_load_pd( sin     );
            sin1 = _mm256_load_pd( sin + 4 );
            rin0 = _mm256_load_pd( real_in     );
            rin1 = _mm256_load_pd( real_in + 4 );
            iin0 = _mm256_load_pd( imag_in     );
            iin1 = _mm256_load_pd( imag_in + 4 );

            tmp0 = _mm256_mul_pd( cos0, rin0 );
            tmp1 = _mm256_mul_pd( cos1, rin1 );
            tmp0 = _mm256_fnmadd_pd( sin0, iin0, tmp0 );
            tmp1 = _mm256_fnmadd_pd( sin1, iin1, tmp1 );
            tmp0 = _mm256_mul_pd( scale0, tmp0 );
            tmp1 = _mm256_mul_pd( scale1, tmp1 );
            _mm256_store_pd( real_out    , tmp0 );
            _mm256_store_pd( real_out + 4, tmp1 );

            tmp0 = _mm256_mul_pd( sin0, rin0 );
            tmp1 = _mm256_mul_pd( sin1, rin1 );
            tmp0 = _mm256_fmadd_pd( cos0, iin0, tmp0 );
            tmp1 = _mm256_fmadd_pd( cos1, iin1, tmp1 );
            tmp0 = _mm256_mul_pd( scale0, tmp0 );
            tmp1 = _mm256_mul_pd( scale1, tmp1 );
            _mm256_store_pd( imag_out    , tmp0 );
            _mm256_store_pd( imag_out + 4, tmp1 );

            cos      = cos      + 8;
            sin      = sin      + 8;
            real_in  = real_in  + 8;
            real_out = real_out + 8;
            imag_in  = imag_in  + 8;
            imag_out = imag_out + 8;
        }
    }
    else
    {
        while ( k-- )
        {
            cos0 = _mm256_load_pd( cos     );
            cos1 = _mm256_load_pd( cos + 4 );
            sin0 = _mm256_load_pd( sin     );
            sin1 = _mm256_load_pd( sin + 4 );
            rin0 = _mm256_load_pd( real_in     );
            rin1 = _mm256_load_pd( real_in + 4 );
            iin0 = _mm256_load_pd( imag_in     );
            iin1 = _mm256_load_pd( imag_in + 4 );

            tmp0 = _mm256_mul_pd( cos0, rin0 );
            tmp1 = _mm256_mul_pd( cos1, rin1 );
            tmp0 = _mm256_fmadd_pd( sin0, iin0, tmp0 );
            tmp1 = _mm256_fmadd_pd( sin1, iin1, tmp1 );
            tmp0 = _mm256_mul_pd( scale0, tmp0 );
            tmp1 = _mm256_mul_pd( scale1, tmp1 );
            _mm256_store_pd( real_out    , tmp0 );
            _mm256_store_pd( real_out + 4, tmp1 );

            tmp0 = _mm256_mul_pd( sin0, rin0 );
            tmp1 = _mm256_mul_pd( sin1, rin1 );
            tmp0 = _mm256_fmsub_pd( cos0, iin0, tmp0 );
            tmp1 = _mm256_fmsub_pd( cos1, iin1, tmp1 );
            tmp0 = _mm256_mul_pd( scale0, tmp0 );
            tmp1 = _mm256_mul_pd( scale1, tmp1 );
            _mm256_store_pd( imag_out    , tmp0 );
            _mm256_store_pd( imag_out + 4, tmp1 );

            cos      = cos      + 8;
            sin      = sin      + 8;
            real_in  = real_in  + 8;
            real_out = real_out + 8;
            imag_in  = imag_in  + 8;
            imag_out = imag_out + 8;
        }
    }
}

void microkernel_double_avx::
swap( const double *mat, const double *in,
            double *out, size_t k, bool pattern ) const noexcept
{
    bool k_odd = k &  1;
         k     = k >> 1;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorpd   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorpd   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorpd   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorpd   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorpd   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorpd   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorpd   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorpd   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorpd   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorpd   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorpd   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        "testq    %[k], %[k]                     \n\t"
        "jz       .Lcheckodd%=                   \n\t"
        "                                        \n\t"
        ".align 16                               \n\t"
        ".Lloop%=:                               \n\t"
        "                                        \n\t"
        "vmovapd    (%[in]), %%ymm12             \n\t" // Even iteration.
        "vmovapd  32(%[in]), %%ymm13             \n\t"
        "                                        \n\t"
        "vbroadcastsd  (%[mat]),%%ymm14          \n\t"
        "vbroadcastsd 8(%[mat]),%%ymm15          \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm0     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm1     \n\t"
        "vfmadd231pd  %%ymm12,%%ymm15,%%ymm4     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm15,%%ymm5     \n\t"
        "                                        \n\t"
        "vbroadcastsd 16(%[mat]),%%ymm14         \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm8     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm9     \n\t"
        "                                        \n\t"
        "vmovapd   64(%[in]), %%ymm12            \n\t" // Odd iteration.
        "vmovapd   96(%[in]), %%ymm13            \n\t"
        "                                        \n\t"
        "vbroadcastsd 24(%[mat]),%%ymm14         \n\t"
        "vbroadcastsd 32(%[mat]),%%ymm15         \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm2     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm3     \n\t"
        "vfmadd231pd  %%ymm12,%%ymm15,%%ymm6     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm15,%%ymm7     \n\t"
        "                                        \n\t"
        "vbroadcastsd 40(%[mat]),%%ymm14         \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm10    \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm11    \n\t"
        "                                        \n\t"
        "addq    $128,%[in]                      \n\t"
        "addq     $48,%[mat]                     \n\t"
        "decq         %[k]                       \n\t"
        "jnz          .Lloop%=                   \n\t"
        "                                        \n\t"
        ".Lcheckodd%=:                           \n\t" // Remaining even iteration, if applicable.
        "testb        %[k_odd],%[k_odd]          \n\t"
        "jz           .Lstoreresult%=            \n\t"
        "vmovapd    (%[in]), %%ymm12             \n\t" // Even iteration.
        "vmovapd  32(%[in]), %%ymm13             \n\t"
        "                                        \n\t"
        "vbroadcastsd  (%[mat]),%%ymm14          \n\t"
        "vbroadcastsd 8(%[mat]),%%ymm15          \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm0     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm1     \n\t"
        "vfmadd231pd  %%ymm12,%%ymm15,%%ymm4     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm15,%%ymm5     \n\t"
        "                                        \n\t"
        "vbroadcastsd 16(%[mat]),%%ymm14         \n\t"
        "vfmadd231pd  %%ymm12,%%ymm14,%%ymm8     \n\t"
        "vfmadd231pd  %%ymm13,%%ymm14,%%ymm9     \n\t"
        "                                        \n\t"
        "                                        \n\t"
        ".Lstoreresult%=:                        \n\t"
        "testb %[pattern],%[pattern]             \n\t"
        "jz    .Lnegativepattern%=               \n\t"
        "vmovapd     %%ymm0,    (%[out])         \n\t"
        "vmovapd     %%ymm1,  32(%[out])         \n\t"
        "vmovapd     %%ymm2,  64(%[out])         \n\t"
        "vmovapd     %%ymm3,  96(%[out])         \n\t"
        "vmovapd     %%ymm4, 128(%[out])         \n\t"
        "vmovapd     %%ymm5, 160(%[out])         \n\t"
        "vmovapd     %%ymm6, 192(%[out])         \n\t"
        "vmovapd     %%ymm7, 224(%[out])         \n\t"
        "vmovapd     %%ymm8, 256(%[out])         \n\t"
        "vmovapd     %%ymm9, 288(%[out])         \n\t"
        "vmovapd     %%ymm10,320(%[out])         \n\t"
        "vmovapd     %%ymm11,352(%[out])         \n\t"
        "jmp .Lfinish%=                          \n\t"
        "                                        \n\t"
        ".Lnegativepattern%=:                    \n\t"
        "vmovapd     %%ymm2,    (%[out])         \n\t"
        "vmovapd     %%ymm3,  32(%[out])         \n\t"
        "vmovapd     %%ymm0,  64(%[out])         \n\t"
        "vmovapd     %%ymm1,  96(%[out])         \n\t"
        "vmovapd     %%ymm6, 128(%[out])         \n\t"
        "vmovapd     %%ymm7, 160(%[out])         \n\t"
        "vmovapd     %%ymm4, 192(%[out])         \n\t"
        "vmovapd     %%ymm5, 224(%[out])         \n\t"
        "vmovapd     %%ymm10,256(%[out])         \n\t"
        "vmovapd     %%ymm11,288(%[out])         \n\t"
        "vmovapd     %%ymm8, 320(%[out])         \n\t"
        "vmovapd     %%ymm9, 352(%[out])         \n\t"
        "                                        \n\t"
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
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}

void microkernel_double_avx::
zm2l( const double* fac, const double* in,
            double* out, size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;
    const double signs { -0.0 };

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorpd   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorpd   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorpd   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorpd   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorpd   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorpd   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorpd   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorpd   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorpd   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorpd   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorpd   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" // do
        "                                        \n\t" // {
        "vmovapd          (%[in]), %%ymm12       \n\t" // b0   = _mm256_load_pd(in+0);
        "vmovapd        32(%[in]), %%ymm13       \n\t" // b1   = _mm256_load_pd(in+4);
        "                                        \n\t"
        "vbroadcastsd    (%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+0); 
        "vbroadcastsd   8(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+1);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm0  \n\t" // c00  = _mm256_fmadd_pd(b0,tmp0,c00)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm1  \n\t" // c01  = _mm256_fmadd_pd(b1,tmp0,c01)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm2  \n\t" // c10  = _mm256_fmadd_pd(b0,tmp1,c10)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm3  \n\t" // c11  = _mm256_fmadd_pd(b1,tmp1,c11)
        "                                        \n\t"
        "vbroadcastsd  16(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+2); 
        "vbroadcastsd  24(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+3);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm4  \n\t" // c20  = _mm256_fmadd_pd(b0,tmp0,c20)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm5  \n\t" // c21  = _mm256_fmadd_pd(b1,tmp0,c21)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm6  \n\t" // c30  = _mm256_fmadd_pd(b0,tmp1,c30)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm7  \n\t" // c31  = _mm256_fmadd_pd(b1,tmp1,c31)
        "                                        \n\t"
        "vbroadcastsd  32(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+4); 
        "vbroadcastsd  40(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+5);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm8  \n\t" // c40  = _mm256_fmadd_pd(b0,tmp0,c40)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm9  \n\t" // c41  = _mm256_fmadd_pd(b1,tmp0,c41)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm10 \n\t" // c50  = _mm256_fmadd_pd(b0,tmp1,c50)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm11 \n\t" // c51  = _mm256_fmadd_pd(b1,tmp1,c51)
        "                                        \n\t"
        "addq          $64,%[in]                 \n\t" // in   = in + 8;
        "addq          $8, %[fac]                \n\t" // fac = fac + 1;
        "decq          %[k]                      \n\t" // --k;
        "jnz           .Lloop%=                  \n\t" // } while (k);
        "                                        \n\t"
        "vbroadcastsd  %[signs], %%ymm15         \n\t" // tmp1 = _mm256_set1_pd( -0.0 );
        "testb         %[pattern], %[pattern]    \n\t" // if ( pattern )
        "jz           .Lnegativepattern%=        \n\t" // {
        "vxorpd        %%ymm15, %%ymm0, %%ymm0   \n\t" // c00 = _mm256_xor_pd( c00, tmp1 )
        "vxorpd        %%ymm15, %%ymm1, %%ymm1   \n\t" // c01 = _mm256_xor_pd( c01, tmp1 )
        "vxorpd        %%ymm15, %%ymm4, %%ymm4   \n\t" // c20 = _mm256_xor_pd( c20, tmp1 )
        "vxorpd        %%ymm15, %%ymm5, %%ymm5   \n\t" // c21 = _mm256_xor_pd( c21, tmp1 )
        "vxorpd        %%ymm15, %%ymm8, %%ymm8   \n\t" // c40 = _mm256_xor_pd( c40, tmp1 )
        "vxorpd        %%ymm15, %%ymm9, %%ymm9   \n\t" // c41 = _mm256_xor_pd( c41, tmp1 )
        "jmp           .Lstoreresult%=           \n\t" // } 
        "                                        \n\t" // else
        ".Lnegativepattern%=:                    \n\t" // {
        "vxorpd        %%ymm15, %%ymm2 , %%ymm2  \n\t" // c10 = _mm256_xor_pd( c10, tmp1 )
        "vxorpd        %%ymm15, %%ymm3 , %%ymm3  \n\t" // c11 = _mm256_xor_pd( c11, tmp1 )
        "vxorpd        %%ymm15, %%ymm6 , %%ymm6  \n\t" // c30 = _mm256_xor_pd( c30, tmp1 )
        "vxorpd        %%ymm15, %%ymm7 , %%ymm7  \n\t" // c31 = _mm256_xor_pd( c31, tmp1 )
        "vxorpd        %%ymm15, %%ymm10, %%ymm10 \n\t" // c50 = _mm256_xor_pd( c50, tmp1 )
        "vxorpd        %%ymm15, %%ymm11, %%ymm11 \n\t" // c51 = _mm256_xor_pd( c51, tmp1 )
        "                                        \n\t" // }
        ".Lstoreresult%=:                        \n\t"
        "vmovapd       %%ymm0,     (%[out])      \n\t" // _mm256_store_pd( out   , c00 );
        "vmovapd       %%ymm1,   32(%[out])      \n\t" // _mm256_store_pd( out+ 4, c01 );
        "vmovapd       %%ymm2,   64(%[out])      \n\t" // _mm256_store_pd( out+ 8, c10 );
        "vmovapd       %%ymm3,   96(%[out])      \n\t" // _mm256_store_pd( out+12, c11 );
        "vmovapd       %%ymm4,  128(%[out])      \n\t" // _mm256_store_pd( out+16, c20 );
        "vmovapd       %%ymm5,  160(%[out])      \n\t" // _mm256_store_pd( out+20, c21 );
        "vmovapd       %%ymm6,  192(%[out])      \n\t" // _mm256_store_pd( out+24, c30 );
        "vmovapd       %%ymm7,  224(%[out])      \n\t" // _mm256_store_pd( out+28, c31 );
        "vmovapd       %%ymm8,  256(%[out])      \n\t" // _mm256_store_pd( out+32, c40 );
        "vmovapd       %%ymm9,  288(%[out])      \n\t" // _mm256_store_pd( out+36, c41 );
        "vmovapd       %%ymm10, 320(%[out])      \n\t" // _mm256_store_pd( out+40, c50 );
        "vmovapd       %%ymm11, 352(%[out])      \n\t" // _mm256_store_pd( out+44, c51 );
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out),
          [pattern] "r"(pattern),
          [signs]   "m"(signs)

        : // Clobbered registers
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}

void microkernel_double_avx::
zm2m( const double* fac, const double* in,
            double* out, size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                        \n\t"
        "vxorpd   %%ymm0 ,%%ymm0, %%ymm0         \n\t"
        "vxorpd   %%ymm1 ,%%ymm1, %%ymm1         \n\t"
        "vxorpd   %%ymm2 ,%%ymm2, %%ymm2         \n\t"
        "vxorpd   %%ymm3 ,%%ymm3, %%ymm3         \n\t"
        "vxorpd   %%ymm4 ,%%ymm4, %%ymm4         \n\t"
        "vxorpd   %%ymm5 ,%%ymm5, %%ymm5         \n\t"
        "vxorpd   %%ymm6 ,%%ymm6, %%ymm6         \n\t"
        "vxorpd   %%ymm7 ,%%ymm7, %%ymm7         \n\t"
        "vxorpd   %%ymm8 ,%%ymm8, %%ymm8         \n\t"
        "vxorpd   %%ymm9 ,%%ymm9, %%ymm9         \n\t"
        "vxorpd   %%ymm10,%%ymm10,%%ymm10        \n\t"
        "vxorpd   %%ymm11,%%ymm11,%%ymm11        \n\t"
        "                                        \n\t"
        ".align 16                               \n\t" 
        ".Lloop%=:                               \n\t" // do
        "                                        \n\t" // {
        "vmovapd          (%[in]), %%ymm12       \n\t" // b0   = _mm256_load_pd(in+0);
        "vmovapd        32(%[in]), %%ymm13       \n\t" // b1   = _mm256_load_pd(in+4);
        "                                        \n\t"
        "vbroadcastsd    (%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+0); 
        "vbroadcastsd   8(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+1);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm0  \n\t" // c50  = _mm256_fmadd_pd(b0,tmp0,c50)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm1  \n\t" // c51  = _mm256_fmadd_pd(b1,tmp0,c51)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm2  \n\t" // c40  = _mm256_fmadd_pd(b0,tmp1,c40)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm3  \n\t" // c41  = _mm256_fmadd_pd(b1,tmp1,c41)
        "                                        \n\t"
        "vbroadcastsd  16(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+2); 
        "vbroadcastsd  24(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+3);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm4  \n\t" // c30  = _mm256_fmadd_pd(b0,tmp0,c30)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm5  \n\t" // c31  = _mm256_fmadd_pd(b1,tmp0,c31)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm6  \n\t" // c20  = _mm256_fmadd_pd(b0,tmp1,c20)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm7  \n\t" // c21  = _mm256_fmadd_pd(b1,tmp1,c21)
        "                                        \n\t"
        "vbroadcastsd  32(%[fac]), %%ymm14       \n\t" // tmp0 = _mm256_broadcast_sd(fac+4); 
        "vbroadcastsd  40(%[fac]), %%ymm15       \n\t" // tmp1 = _mm256_broadcast_sd(fac+5);
        "vfmadd231pd   %%ymm12, %%ymm14, %%ymm8  \n\t" // c10  = _mm256_fmadd_pd(b0,tmp0,c10)
        "vfmadd231pd   %%ymm13, %%ymm14, %%ymm9  \n\t" // c11  = _mm256_fmadd_pd(b1,tmp0,c11)
        "vfmadd231pd   %%ymm12, %%ymm15, %%ymm10 \n\t" // c00  = _mm256_fmadd_pd(b0,tmp1,c00)
        "vfmadd231pd   %%ymm13, %%ymm15, %%ymm11 \n\t" // c01  = _mm256_fmadd_pd(b1,tmp1,c01)
        "                                        \n\t"
        "addq          $64,%[in]                 \n\t" // in   = in + 8;
        "addq          $8, %[fac]                \n\t" // fac = fac + 1;
        "decq          %[k]                      \n\t" // --k;
        "jnz           .Lloop%=                  \n\t" // } while (k);
        "                                        \n\t"
        "vmovapd       %%ymm10,    (%[out])      \n\t" // _mm256_store_pd( out   , c00 );
        "vmovapd       %%ymm11,  32(%[out])      \n\t" // _mm256_store_pd( out+ 4, c01 );
        "vmovapd       %%ymm8,   64(%[out])      \n\t" // _mm256_store_pd( out+ 8, c10 );
        "vmovapd       %%ymm9,   96(%[out])      \n\t" // _mm256_store_pd( out+12, c11 );
        "vmovapd       %%ymm6,  128(%[out])      \n\t" // _mm256_store_pd( out+16, c20 );
        "vmovapd       %%ymm7,  160(%[out])      \n\t" // _mm256_store_pd( out+20, c21 );
        "vmovapd       %%ymm4,  192(%[out])      \n\t" // _mm256_store_pd( out+24, c30 );
        "vmovapd       %%ymm5,  224(%[out])      \n\t" // _mm256_store_pd( out+28, c31 );
        "vmovapd       %%ymm2,  256(%[out])      \n\t" // _mm256_store_pd( out+32, c40 );
        "vmovapd       %%ymm3,  288(%[out])      \n\t" // _mm256_store_pd( out+36, c41 );
        "vmovapd       %%ymm0,  320(%[out])      \n\t" // _mm256_store_pd( out+40, c50 );
        "vmovapd       %%ymm1,  352(%[out])      \n\t" // _mm256_store_pd( out+44, c51 );
 
        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [k]       "+r"(k)

        : // Input  operands
          [out]     "r"(out)

        : // Clobbered registers
          "ymm0" , "ymm1" , "ymm2" , "ymm3" ,
          "ymm4" , "ymm5" , "ymm6" , "ymm7" ,
          "ymm8" , "ymm9" , "ymm10", "ymm11",
          "ymm12", "ymm13", "ymm14", "ymm15"
    );
}

void microkernel_double_avx::
swap2trans_buf( const double *__restrict__ real_in, 
                const double *__restrict__ imag_in,
                      double **real_out, 
                      double **imag_out,
                      size_t n ) const noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        double *__restrict__ rdst = real_out[m] + (n-m)*8;
        double *__restrict__ idst = imag_out[m] + (n-m)*8;

        __m256d rtmp0 = _mm256_load_pd( real_in     );
        __m256d rtmp1 = _mm256_load_pd( real_in + 4 );
        __m256d itmp0 = _mm256_load_pd( imag_in     );
        __m256d itmp1 = _mm256_load_pd( imag_in + 4 );
        _mm256_store_pd( rdst    , rtmp0 );
        _mm256_store_pd( rdst + 4, rtmp1 );
        _mm256_store_pd( idst    , itmp0 );
        _mm256_store_pd( idst + 4, itmp1 );
        real_in += 8;
        imag_in += 8;
    }
}

void microkernel_double_avx::
trans2swap_buf( const double *const *const real_in, 
                const double *const *const imag_in,
                      double *__restrict__ real_out, 
                      double *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    using std::min;
    __m256d zeros = _mm256_setzero_pd();

    for ( size_t m = 0; m <= min(n,Pmax); ++m )
    {
        const double *__restrict__ rsrc = real_in[m] + (n-m)*8;
        const double *__restrict__ isrc = imag_in[m] + (n-m)*8;
        __m256d rtmp0 = _mm256_load_pd( rsrc     );
        __m256d rtmp1 = _mm256_load_pd( rsrc + 4 );
        __m256d itmp0 = _mm256_load_pd( isrc     );
        __m256d itmp1 = _mm256_load_pd( isrc + 4 );
        _mm256_store_pd( real_out    , rtmp0 );
        _mm256_store_pd( real_out + 4, rtmp1 );
        _mm256_store_pd( imag_out    , itmp0 );
        _mm256_store_pd( imag_out + 4, itmp1 );
        real_out += 8;
        imag_out += 8;
    }

    for ( size_t m = min(n,Pmax); m <= n; ++m )
    {
        _mm256_store_pd( real_out    , zeros );
        _mm256_store_pd( real_out + 4, zeros );
        _mm256_store_pd( imag_out    , zeros );
        _mm256_store_pd( imag_out + 4, zeros );
        real_out += 8;
        imag_out += 8;
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

void microkernel_double_avx::
buf2solid( const double  *real_in,  const double *imag_in, 
                 double **p_solids,       double *trash, const size_t *P, size_t n ) const noexcept
{
    double *solids[ 8 ] =
    {
        ( n < P[0] ) ? p_solids[0] : trash,
        ( n < P[1] ) ? p_solids[1] : trash,
        ( n < P[2] ) ? p_solids[2] : trash,
        ( n < P[3] ) ? p_solids[3] : trash,
        ( n < P[4] ) ? p_solids[4] : trash,
        ( n < P[5] ) ? p_solids[5] : trash,
        ( n < P[6] ) ? p_solids[6] : trash,
        ( n < P[7] ) ? p_solids[7] : trash
    };

    size_t m;
    __m256d r00, r01, r10, r11, r20, r21, r30, r31;
    for ( m = 0; m + 3 <= n; m += 4 )
    {
        r00 = _mm256_load_pd( real_in + (m+0)*8 + 0 );
        r01 = _mm256_load_pd( real_in + (m+0)*8 + 4 );
        r10 = _mm256_load_pd( real_in + (m+1)*8 + 0 );
        r11 = _mm256_load_pd( real_in + (m+1)*8 + 4 );
        r20 = _mm256_load_pd( real_in + (m+2)*8 + 0 );
        r21 = _mm256_load_pd( real_in + (m+2)*8 + 4 );
        r30 = _mm256_load_pd( real_in + (m+3)*8 + 0 );
        r31 = _mm256_load_pd( real_in + (m+3)*8 + 4 );
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

        r00 = _mm256_load_pd( imag_in + (m+0)*8 + 0 );
        r01 = _mm256_load_pd( imag_in + (m+0)*8 + 4 );
        r10 = _mm256_load_pd( imag_in + (m+1)*8 + 0 );
        r11 = _mm256_load_pd( imag_in + (m+1)*8 + 4 );
        r20 = _mm256_load_pd( imag_in + (m+2)*8 + 0 );
        r21 = _mm256_load_pd( imag_in + (m+2)*8 + 4 );
        r30 = _mm256_load_pd( imag_in + (m+3)*8 + 0 );
        r31 = _mm256_load_pd( imag_in + (m+3)*8 + 4 );
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
    }

    // Copy the remaining parts in scalar mode.
    for ( size_t l  = 0; l  <  8; ++l  )
    for ( size_t mm = m; mm <= n; ++mm )
    {
        solids[l][ (n  )*(n+1) + mm ] += real_in[ mm*8 + l ];
        solids[l][ (n+1)*(n+1) + mm ] += imag_in[ mm*8 + l ];
    } 
}

void microkernel_double_avx::
solid2buf( const double *const *p_solids, const double *const zeros, const size_t *P,
                 double *real_out, double *imag_out, size_t n ) const noexcept
{
    const double *const solids[ 8 ] =
    {
        ( n < P[0] ) ? p_solids[0] : zeros,
        ( n < P[1] ) ? p_solids[1] : zeros,
        ( n < P[2] ) ? p_solids[2] : zeros,
        ( n < P[3] ) ? p_solids[3] : zeros,
        ( n < P[4] ) ? p_solids[4] : zeros,
        ( n < P[5] ) ? p_solids[5] : zeros,
        ( n < P[6] ) ? p_solids[6] : zeros,
        ( n < P[7] ) ? p_solids[7] : zeros
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
        _mm256_store_pd( real_out + (m+0)*8 + 0, r00 );
        _mm256_store_pd( real_out + (m+0)*8 + 4, r01 );
        _mm256_store_pd( real_out + (m+1)*8 + 0, r10 );
        _mm256_store_pd( real_out + (m+1)*8 + 4, r11 );
        _mm256_store_pd( real_out + (m+2)*8 + 0, r20 );
        _mm256_store_pd( real_out + (m+2)*8 + 4, r21 );
        _mm256_store_pd( real_out + (m+3)*8 + 0, r30 );
        _mm256_store_pd( real_out + (m+3)*8 + 4, r31 );

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
        _mm256_store_pd( imag_out + (m+0)*8 + 0, r00 );
        _mm256_store_pd( imag_out + (m+0)*8 + 4, r01 );
        _mm256_store_pd( imag_out + (m+1)*8 + 0, r10 );
        _mm256_store_pd( imag_out + (m+1)*8 + 4, r11 );
        _mm256_store_pd( imag_out + (m+2)*8 + 0, r20 );
        _mm256_store_pd( imag_out + (m+2)*8 + 4, r21 );
        _mm256_store_pd( imag_out + (m+3)*8 + 0, r30 );
        _mm256_store_pd( imag_out + (m+3)*8 + 4, r31 );
    }

    // Remaining part in scalar mode.
    for ( size_t l  = 0; l  <  8; ++l  )
    for ( size_t mm = m; mm <= n; ++mm )
    {
        real_out[ mm*8 + l ] = solids[l][ (n  )*(n+1) + mm ];
        imag_out[ mm*8 + l ] = solids[l][ (n+1)*(n+1) + mm ];
    }
}

#undef transpose

}

#ifdef __llvm__
#pragma clang attribute pop
#endif

#endif // Check for amd64

