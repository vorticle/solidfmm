/*
 * Copyright (C) 2022 Matthias Kirchhart
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
#include <solidfmm/microkernel_armv8a.hpp>

#ifdef SOLIDFMM_ARMv8A

// For reasons beyond human comprehension, the LLVM assembler deviates from the
// standard ARMv8 syntax for a few instructions.
#ifdef __llvm__
#define DFMLA(a,b,c,n) " fmla.2d "#a","#b","#c"["#n"]         \n\t"
#define DFMLAV(a,b,c)  " fmla.2d "#a","#b","#c"               \n\t"
#define DFMLSV(a,b,c)  " fmls.2d "#a","#b","#c"               \n\t"
#define DFMUL(a,b,c)   " fmul.2d "#a","#b","#c"               \n\t"
#define DFADD(a,b,c)   " fadd.2d "#a","#b","#c"               \n\t"
#define DFDIV(a,b,c)   " fdiv.2d "#a","#b","#c"               \n\t"
#define DFSQRT(a,b)    " fsqrt.2d "#a","#b"                   \n\t"
#define DFCMEQZ(a,b)   " fcmeq.2d "#a","#b", #0.0             \n\t"
#define DFNEG(a,b)     " fneg.2d  "#a","#b"                   \n\t"
#define LOAD64(a,b,c,d,e)     " ld1.16b { "#a", "#b", "#c", "#d" }, ["#e"]      \n\t"
#define LOAD64inc(a,b,c,d,e)  " ld1.16b { "#a", "#b", "#c", "#d" }, ["#e"], #64 \n\t"
#define LOAD48inc(a,b,c,d)    " ld1.16b { "#a", "#b", "#c" }, ["#d"], #48       \n\t"
#define STORE64(a,b,c,d,e)    " st1.16b { "#a", "#b", "#c", "#d" }, ["#e"]      \n\t"
#define STORE64inc(a,b,c,d,e) " st1.16b { "#a", "#b", "#c", "#d" }, ["#e"], #64 \n\t"
#define DST4inc(a,b,c,d,e,f)  " st4.d   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #32 \n\t"
#define DLD4inc(a,b,c,d,e,f)  " ld4.d   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #32 \n\t"
#else
#define DFMLA(a,b,c,n) " fmla    "#a".2d,"#b".2d,"#c".d["#n"] \n\t"
#define DFMLAV(a,b,c)  " fmla    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define DFMLSV(a,b,c)  " fmls    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define DFMUL(a,b,c)   " fmul    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define DFADD(a,b,c)   " fadd    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define DFDIV(a,b,c)   " fdiv    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define DFSQRT(a,b)    " fsqrt   "#a".2d,"#b".2d              \n\t"
#define DFCMEQZ(a,b)   " fcmeq   "#a".2d,"#b".2d, #0.0        \n\t"
#define DFNEG(a,b)     " fneg    "#a".2d,"#b".2d              \n\t"
#define LOAD64(a,b,c,d,e)     " ld1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"]      \n\t"
#define LOAD64inc(a,b,c,d,e)  " ld1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"], #64 \n\t"
#define LOAD48inc(a,b,c,d)    " ld1 { "#a".16b, "#b".16b, "#c".16b }, ["#d"], #48           \n\t"
#define STORE64(a,b,c,d,e)    " st1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"]      \n\t"
#define STORE64inc(a,b,c,d,e) " st1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"], #64 \n\t"
#define DST4inc(a,b,c,d,e,f)  " st4 { "#a".d,   "#b".d,   "#c".d,   "#d".d  }["#e"], ["#f"], #32 \n\t"
#define DLD4inc(a,b,c,d,e,f)  " ld4 { "#a".d,   "#b".d,   "#c".d,   "#d".d  }["#e"], ["#f"], #32 \n\t"
#endif

namespace solidfmm
{

microkernel_double_armv8a::microkernel_double_armv8a():
microkernel<double> { 6, 8, 64 }
{}

void microkernel_double_armv8a::
euler( const double *x,   const double *y,   const double *z,
             double *r,         double *rinv,
             double *cos_alpha, double *sin_alpha,
             double *cos_beta,  double *sin_beta,
             size_t k ) const noexcept
{
    __asm__ volatile
    (
        "                                   \n\t"
        " cbz %[k], 9f                      \n\t" // if ( k == 0 ) return;
        "                                   \n\t"
        " eor v0.16b, v0.16b, v0.16b        \n\t" // v0 contains 0.0
        " eor v1.16b, v1.16b, v1.16b        \n\t" // v1 contains 0.0
        " eor v2.16b, v2.16b, v2.16b        \n\t" // v2 contains 0.0
        " eor v3.16b, v3.16b, v3.16b        \n\t" // v3 contains 0.0
        "                                   \n\t"
        " mov x12, #1                       \n\t" // Integer 1 into x12
        " dup v4.2d, x12                    \n\t" // x12 into v4.d[0] and v4.d[1]
        " ucvtf v4.2d, v4.2d                \n\t" // Convert v4 into floating point
        " fmov x12, d4                      \n\t" // v4 and x12 now contain the value 1.0
        " mov v5.16b, v4.16b                \n\t" // v4-v7 and x12 now contain floating point 1.0.
        " mov v6.16b, v4.16b                \n\t"
        " mov v7.16b, v4.16b                \n\t"

        STORE64inc(v4,v5,v6,v7,%[r])
        STORE64inc(v4,v5,v6,v7,%[rinv])
        STORE64inc(v4,v5,v6,v7,%[cos_alpha])
        STORE64inc(v0,v1,v2,v3,%[sin_alpha])
        STORE64inc(v4,v5,v6,v7,%[cos_beta])
        STORE64inc(v0,v1,v2,v3,%[sin_beta])
        " subs %[k], %[k], #1               \n\t"
        " cbz %[k], 9f                      \n\t" // if ( k == 1 ) return;
        "                                   \n\t"
        LOAD64(v16,v17,v18,v19,%[x])
        LOAD64(v20,v21,v22,v23,%[y])
        LOAD64(v24,v25,v26,v27,%[z])
        "                                   \n\t"
        DFMUL(v28,v16,v16)
        DFMUL(v29,v17,v17)
        DFMUL(v30,v18,v18)
        DFMUL(v31,v19,v19)
        "                                   \n\t"
        DFMLAV(v28,v20,v20)
        DFMLAV(v29,v21,v21)
        DFMLAV(v30,v22,v22)
        DFMLAV(v31,v23,v23)
        "                                   \n\t"
        DFSQRT(v28,v28)                           // v28-v31 now contain rxy
        DFSQRT(v29,v29)
        DFSQRT(v30,v30)
        DFSQRT(v31,v31)
        "                                   \n\t"
        DFMUL(v0,v24,v24)
        DFMUL(v1,v25,v25)
        DFMUL(v2,v26,v26)
        DFMUL(v3,v27,v27)
        "                                   \n\t"
        DFMLAV(v0,v28,v28)                       
        DFMLAV(v1,v29,v29)
        DFMLAV(v2,v30,v30)
        DFMLAV(v3,v31,v31)
        "                                   \n\t"
        DFSQRT(v0,v0)                             // v0-v3 now contain r = rxyz
        DFSQRT(v1,v1)
        DFSQRT(v2,v2)
        DFSQRT(v3,v3)
        "                                   \n\t"
        STORE64inc(v0,v1,v2,v3,%[r])
        "                                   \n\t"
        " dup v4.2d, x12                    \n\t" 
        " dup v5.2d, x12                    \n\t"
        " dup v6.2d, x12                    \n\t"
        " dup v7.2d, x12                    \n\t"
        "                                   \n\t"
        DFDIV(v4,v4,v0)                           // v4-v7 now contain 1/r
        DFDIV(v5,v5,v1)  
        DFDIV(v6,v6,v2)  
        DFDIV(v7,v7,v3)  
        "                                   \n\t"
        STORE64inc(v4,v5,v6,v7,%[rinv])
        "                                   \n\t"
        DFCMEQZ(v8,v28)                           // rxy == 0 ?
        DFCMEQZ(v9,v29) 
        DFCMEQZ(v10,v30) 
        DFCMEQZ(v11,v31) 
        "                                   \n\t"
        DFDIV(v12,v20,v28)                        // cos_alpha = y/rxy
        DFDIV(v13,v21,v29)                        // no need for y beyond this point.
        DFDIV(v14,v22,v30)   
        DFDIV(v15,v23,v31)   
        "                                   \n\t"
        " bic v12.16b, v12.16b, v8.16b      \n\t" // cos_alpha = 0 if rxy == 0 
        " bic v13.16b, v13.16b, v9.16b      \n\t" 
        " bic v14.16b, v14.16b, v10.16b     \n\t" 
        " bic v15.16b, v15.16b, v11.16b     \n\t" 
        "                                   \n\t"
        " dup v20.2d, x12                   \n\t" // We may overwrite the previous y-register
        " dup v21.2d, x12                   \n\t" // with 1.0.
        " dup v22.2d, x12                   \n\t"
        " dup v23.2d, x12                   \n\t"
        "                                   \n\t"
        " and v20.16b, v20.16b, v8.16b      \n\t" // Contains 1.0 where rxy == 0
        " and v21.16b, v21.16b, v9.16b      \n\t" 
        " and v22.16b, v22.16b, v10.16b     \n\t"
        " and v23.16b, v23.16b, v11.16b     \n\t"
        "                                   \n\t"
        DFADD(v12,v12,v20)                        // cos_alpha = ( rxy == 0 ) ? 1 : y/rxy
        DFADD(v13,v13,v21)
        DFADD(v14,v14,v22)
        DFADD(v15,v15,v23)
        "                                   \n\t"
        STORE64(v12,v13,v14,v15,%[cos_alpha])
        "                                   \n\t"
        DFDIV(v12,v16,v28)                        // sin_alpha = x/rxy
        DFDIV(v13,v17,v29)                        // no need for x beyond this point.
        DFDIV(v14,v18,v30)   
        DFDIV(v15,v19,v31)   
        "                                   \n\t"
        " bic v12.16b, v12.16b, v8.16b      \n\t" // sin_alpha = 0 if rxy == 0 
        " bic v13.16b, v13.16b, v9.16b      \n\t" 
        " bic v14.16b, v14.16b, v10.16b     \n\t" 
        " bic v15.16b, v15.16b, v11.16b     \n\t" 
        "                                   \n\t"
        STORE64(v12,v13,v14,v15,%[sin_alpha])
        "                                   \n\t"
        DFNEG(v28,v28)                            // rxy = -rxy.
        DFNEG(v29,v29)
        DFNEG(v30,v30)
        DFNEG(v31,v31)
        "                                   \n\t"
        DFMUL(v24,v24,v4)                         // cos_beta = z/r
        DFMUL(v25,v25,v5)                   
        DFMUL(v26,v26,v6)                   
        DFMUL(v27,v27,v7)                   
        "                                   \n\t"
        DFMUL(v28,v28,v4)                         // sin_beta = -rxy/r
        DFMUL(v29,v29,v5)                   
        DFMUL(v30,v30,v6)                   
        DFMUL(v31,v31,v7)                   
        "                                   \n\t"
        STORE64(v24,v25,v26,v27,%[cos_beta])
        STORE64(v28,v29,v30,v31,%[sin_beta])
        "                                   \n\t"
        " subs %[k], %[k], #1               \n\t"
        " cbz %[k], 9f                      \n\t" // if ( k == 2 ) return;
        "                                   \n\t"
        " mov x12, %[k]                     \n\t" // Compute powers of r and rinv.
        " mov v8.16b , v0.16b               \n\t"
        " mov v9.16b , v1.16b               \n\t"
        " mov v10.16b, v2.16b               \n\t"
        " mov v11.16b, v3.16b               \n\t"
        " mov v12.16b, v4.16b               \n\t"
        " mov v13.16b, v5.16b               \n\t"
        " mov v14.16b, v6.16b               \n\t"
        " mov v15.16b, v7.16b               \n\t"

        " .align 5                          \n\t" 
        " 1:                                \n\t"
        DFMUL(v8,v8,v0)
        DFMUL(v9,v9,v1)
        DFMUL(v10,v10,v2)
        DFMUL(v11,v11,v3)

        DFMUL(v12,v12,v4)
        DFMUL(v13,v13,v5)
        DFMUL(v14,v14,v6)
        DFMUL(v15,v15,v7)

        STORE64inc(v8,v9,v10,v11,%[r])
        STORE64inc(v12,v13,v14,v15,%[rinv])
        " subs x12, x12, #1                 \n\t"
        " b.ne 1b                           \n\t"
        
        
        LOAD64(v0,v1,v2,v3,%[cos_alpha])         // Compute powers of cos_alpha, sin_alpha
        LOAD64(v4,v5,v6,v7,%[sin_alpha])
        LOAD64inc(v8,v9,v10,v11,%[cos_alpha])
        LOAD64inc(v12,v13,v14,v15,%[sin_alpha])
       
        " mov x12, %[k]                     \n\t"
        " .align 5                          \n\t" 
        " 2:                                \n\t"
        " mov v16.16b, v8.16b               \n\t"
        " mov v17.16b, v9.16b               \n\t"
        " mov v18.16b, v10.16b              \n\t"
        " mov v19.16b, v11.16b              \n\t"
        " mov v20.16b, v12.16b              \n\t"
        " mov v21.16b, v13.16b              \n\t"
        " mov v22.16b, v14.16b              \n\t"
        " mov v23.16b, v15.16b              \n\t"

        DFMUL(v8,v16,v0)
        DFMUL(v9,v17,v1)
        DFMUL(v10,v18,v2)
        DFMUL(v11,v19,v3)
        
        DFMUL(v12,v16,v4)
        DFMUL(v13,v17,v5)
        DFMUL(v14,v18,v6)
        DFMUL(v15,v19,v7)

        DFMLSV(v8,v20,v4)
        DFMLSV(v9,v21,v5)
        DFMLSV(v10,v22,v6)
        DFMLSV(v11,v23,v7)
       
        DFMLAV(v12,v20,v0) 
        DFMLAV(v13,v21,v1) 
        DFMLAV(v14,v22,v2) 
        DFMLAV(v15,v23,v3) 

        STORE64inc(v8,v9,v10,v11,%[cos_alpha])
        STORE64inc(v12,v13,v14,v15,%[sin_alpha])
        " subs x12, x12, #1               \n\t"
        " b.ne 2b                         \n\t"
       
        LOAD64(v0,v1,v2,v3,%[cos_beta])         // Compute powers of cos_beta, sin_beta
        LOAD64(v4,v5,v6,v7,%[sin_beta])
        LOAD64inc(v8,v9,v10,v11,%[cos_beta])
        LOAD64inc(v12,v13,v14,v15,%[sin_beta])
        
        " .align 5                          \n\t" 
        " 2:                                \n\t"
        " mov v16.16b, v8.16b               \n\t"
        " mov v17.16b, v9.16b               \n\t"
        " mov v18.16b, v10.16b              \n\t"
        " mov v19.16b, v11.16b              \n\t"
        " mov v20.16b, v12.16b              \n\t"
        " mov v21.16b, v13.16b              \n\t"
        " mov v22.16b, v14.16b              \n\t"
        " mov v23.16b, v15.16b              \n\t"

        DFMUL(v8,v16,v0)
        DFMUL(v9,v17,v1)
        DFMUL(v10,v18,v2)
        DFMUL(v11,v19,v3)
        
        DFMUL(v12,v16,v4)
        DFMUL(v13,v17,v5)
        DFMUL(v14,v18,v6)
        DFMUL(v15,v19,v7)

        DFMLSV(v8,v20,v4)
        DFMLSV(v9,v21,v5)
        DFMLSV(v10,v22,v6)
        DFMLSV(v11,v23,v7)
       
        DFMLAV(v12,v20,v0) 
        DFMLAV(v13,v21,v1) 
        DFMLAV(v14,v22,v2) 
        DFMLAV(v15,v23,v3) 

        STORE64inc(v8,v9,v10,v11,%[cos_beta])
        STORE64inc(v12,v13,v14,v15,%[sin_beta])
        " subs %[k], %[k], #1               \n\t"
        " b.ne 2b                           \n\t"
        " 9:                                \n\t" // End.

        : // Output operands.
          [k]         "+r"(k),
          [r]         "+r"(r),
          [rinv]      "+r"(rinv),
          [cos_alpha] "+r"(cos_alpha),
          [sin_alpha] "+r"(sin_alpha),
          [cos_beta]  "+r"(cos_beta),
          [sin_beta]  "+r"(sin_beta)

        : // Input operands.
          [x]         "r"(x),
          [y]         "r"(y),
          [z]         "r"(z) 

        : // Clobbered registers.
        "x12",
        "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
        "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
          
    );
}

void microkernel_double_armv8a::
rotscale( const double *cos,
          const double *sin,
          const double *scale,
          const double *real_in,
          const double *imag_in,
                double *real_out,
                double *imag_out,
          size_t k, bool forward ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                   \n\t"
        LOAD64(v16,v17,v18,v19,%[scale])
        " cbz %w[forward], 2f               \n\t"
        "                                   \n\t"
        " .align 5                          \n\t"
        " 1:                                \n\t" // Forward rotscale.
          LOAD64inc(v0,v1,v2,v3,%[cos])
          LOAD64inc(v4,v5,v6,v7,%[sin])
          LOAD64inc(v8,v9,v10,v11,%[real_in])
          LOAD64inc(v12,v13,v14,v15,%[imag_in])

          DFMUL(v20,v0,v8)                        // real_out = cos*real_in
          DFMUL(v21,v1,v9)
          DFMUL(v22,v2,v10)
          DFMUL(v23,v3,v11)

          DFMUL(v24,v0,v12)                       // imag_out = cos*imag_in
          DFMUL(v25,v1,v13)
          DFMUL(v26,v2,v14)
          DFMUL(v27,v3,v15)

          DFMLSV(v20,v4,v12)                      // real_out -= sin*imag_in
          DFMLSV(v21,v5,v13)
          DFMLSV(v22,v6,v14)
          DFMLSV(v23,v7,v15)

          DFMLAV(v24,v4,v8)                       // imag_out += sin*real_in
          DFMLAV(v25,v5,v9)
          DFMLAV(v26,v6,v10)
          DFMLAV(v27,v7,v11)

          DFMUL(v20,v20,v16)                       // real_out *= fac
          DFMUL(v21,v21,v17)                       
          DFMUL(v22,v22,v18)                       
          DFMUL(v23,v23,v19)                       

          DFMUL(v24,v24,v16)                       // imag_out *= fac
          DFMUL(v25,v25,v17)                       
          DFMUL(v26,v26,v18)                       
          DFMUL(v27,v27,v19)                       

          STORE64inc(v20,v21,v22,v23,%[real_out])
          STORE64inc(v24,v25,v26,v27,%[imag_out])
        " subs %[k], %[k], 1                \n\t"
        " b.ne 1b                           \n\t"
        " b 3f                              \n\t"
        "                                   \n\t"
        " .align 5                          \n\t" 
        " 2:                                \n\t" // Backward rotscale.
          LOAD64inc(v0,v1,v2,v3,%[cos])
          LOAD64inc(v4,v5,v6,v7,%[sin])
          LOAD64inc(v8,v9,v10,v11,%[real_in])
          LOAD64inc(v12,v13,v14,v15,%[imag_in])
        
          DFMUL(v20,v0,v8)                        // real_out = cos*real_in
          DFMUL(v21,v1,v9)
          DFMUL(v22,v2,v10)
          DFMUL(v23,v3,v11)

          DFMUL(v24,v0,v12)                       // imag_out = cos*imag_in
          DFMUL(v25,v1,v13)
          DFMUL(v26,v2,v14)
          DFMUL(v27,v3,v15)

          DFMLAV(v20,v4,v12)                      // real_out += sin*imag_in
          DFMLAV(v21,v5,v13)
          DFMLAV(v22,v6,v14)
          DFMLAV(v23,v7,v15)

          DFMLSV(v24,v4,v8)                       // imag_out -= sin*real_in
          DFMLSV(v25,v5,v9)
          DFMLSV(v26,v6,v10)
          DFMLSV(v27,v7,v11)

          DFMUL(v20,v20,v16)                       // real_out *= fac
          DFMUL(v21,v21,v17)                       
          DFMUL(v22,v22,v18)                       
          DFMUL(v23,v23,v19)                       

          DFMUL(v24,v24,v16)                       // imag_out *= fac
          DFMUL(v25,v25,v17)                       
          DFMUL(v26,v26,v18)                       
          DFMUL(v27,v27,v19)                       

          STORE64inc(v20,v21,v22,v23,%[real_out])
          STORE64inc(v24,v25,v26,v27,%[imag_out])
        " subs %[k], %[k], 1                \n\t"
        " b.ne 2b                           \n\t"
        "                                   \n\t"
        " 3:                                \n\t"
        : // Output operands.
          [cos]      "+r"(cos),
          [sin]      "+r"(sin),
          [real_in]  "+r"(real_in),
          [imag_in]  "+r"(imag_in),
          [real_out] "+r"(real_out),
          [imag_out] "+r"(imag_out),
          [k]        "+r"(k)

        : // Input operands.
          [scale]    "r"(scale),
          [forward]  "r"(forward)

        : // Clobbered registers.
        "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
        "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27"
    );
}

void microkernel_double_armv8a::
swap( const double *mat,
      const double *in,
            double *out,
      size_t k, bool pattern ) const noexcept
{
    size_t k_odd = k &  1;
           k     = k >> 1;

    __asm__ volatile
    (
        "                                   \n\t"  // Zero the result registers.
        " eor v0.16b , v0.16b , v0.16b      \n\t"  // Row 0 of result
        " eor v1.16b , v1.16b , v1.16b      \n\t"  
        " eor v2.16b , v2.16b , v2.16b      \n\t"  
        " eor v3.16b , v3.16b , v3.16b      \n\t"  
        " eor v4.16b , v4.16b , v4.16b      \n\t"  // Row 1 of result
        " eor v5.16b , v5.16b , v5.16b      \n\t" 
        " eor v6.16b , v6.16b , v6.16b      \n\t"
        " eor v7.16b , v7.16b , v7.16b      \n\t"
        " eor v8.16b , v8.16b , v8.16b      \n\t"  // Row 2 of result
        " eor v9.16b , v9.16b , v9.16b      \n\t"
        " eor v10.16b, v10.16b, v10.16b     \n\t"
        " eor v11.16b, v11.16b, v11.16b     \n\t"
        " eor v12.16b, v12.16b, v12.16b     \n\t"  // Row 3 of result
        " eor v13.16b, v13.16b, v13.16b     \n\t"
        " eor v14.16b, v14.16b, v14.16b     \n\t"
        " eor v15.16b, v15.16b, v15.16b     \n\t"
        " eor v16.16b, v16.16b, v16.16b     \n\t"  // Row 4 of result
        " eor v17.16b, v17.16b, v17.16b     \n\t"
        " eor v18.16b, v18.16b, v18.16b     \n\t"
        " eor v19.16b, v19.16b, v19.16b     \n\t"
        " eor v20.16b, v20.16b, v20.16b     \n\t"  // Row 5 of result
        " eor v21.16b, v21.16b, v21.16b     \n\t"
        " eor v22.16b, v22.16b, v22.16b     \n\t"
        " eor v23.16b, v23.16b, v23.16b     \n\t"
        "                                   \n\t"
        " cbz %[k], 2f                      \n\t"
        "                                   \n\t"
        " .align 5                          \n\t"
        " 1:                                \n\t" // do ...
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next row of input
        LOAD48inc(v28,v29,v30,%[mat])             // Load the next two columns of swap matrix
        "                                   \n\t"
        DFMLA(v0,v24,v28,0)
        DFMLA(v1,v25,v28,0)
        DFMLA(v2,v26,v28,0)
        DFMLA(v3,v27,v28,0)
        "                                   \n\t"
        DFMLA(v8,v24,v28,1)
        DFMLA(v9,v25,v28,1)
        DFMLA(v10,v26,v28,1)
        DFMLA(v11,v27,v28,1)
        "                                   \n\t"
        DFMLA(v16,v24,v29,0)
        DFMLA(v17,v25,v29,0)
        DFMLA(v18,v26,v29,0)
        DFMLA(v19,v27,v29,0)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])        // Load next row of input
        "                                   \n\t"
        DFMLA(v4,v24,v29,1)
        DFMLA(v5,v25,v29,1)
        DFMLA(v6,v26,v29,1)
        DFMLA(v7,v27,v29,1)
        "                                   \n\t"
        DFMLA(v12,v24,v30,0)
        DFMLA(v13,v25,v30,0)
        DFMLA(v14,v26,v30,0)
        DFMLA(v15,v27,v30,0)
        "                                   \n\t"
        DFMLA(v20,v24,v30,1)
        DFMLA(v21,v25,v30,1)
        DFMLA(v22,v26,v30,1)
        DFMLA(v23,v27,v30,1)
        "                                   \n\t"
        " subs %[k], %[k], #1               \n\t"
        " b.ne 1b                           \n\t" // while ( k-- );
        "                                   \n\t"
        " 2:                                \n\t"
        " cbz %[k_odd], 3f                  \n\t" // if ( k_odd )
        LOAD64(v24,v25,v26,v27,%[in])             // Load next row of input
        " ldr q28, [%[mat]]                 \n\t" // Load the last column if swap matrix.
        " ldr d29, [%[mat],#16]             \n\t"
        "                                   \n\t"
        DFMLA(v0,v24,v28,0)
        DFMLA(v1,v25,v28,0)
        DFMLA(v2,v26,v28,0)
        DFMLA(v3,v27,v28,0)
        "                                   \n\t"
        DFMLA(v8,v24,v28,1)
        DFMLA(v9,v25,v28,1)
        DFMLA(v10,v26,v28,1)
        DFMLA(v11,v27,v28,1)
        "                                   \n\t"
        DFMLA(v16,v24,v29,0)
        DFMLA(v17,v25,v29,0)
        DFMLA(v18,v26,v29,0)
        DFMLA(v19,v27,v29,0)
        "                                   \n\t"
        " 3:                                \n\t"
        " cbz %w[pattern], 4f               \n\t" // if ( pattern )
        STORE64inc(v0,v1,v2,v3,%[out])
        STORE64inc(v4,v5,v6,v7,%[out])
        STORE64inc(v8,v9,v10,v11,%[out])
        STORE64inc(v12,v13,v14,v15,%[out])
        STORE64inc(v16,v17,v18,v19,%[out])
        STORE64inc(v20,v21,v22,v23,%[out])
        " b 5f                              \n\t"
        " 4:                                \n\t" // else
        STORE64inc(v4,v5,v6,v7,%[out])
        STORE64inc(v0,v1,v2,v3,%[out])
        STORE64inc(v12,v13,v14,v15,%[out])
        STORE64inc(v8,v9,v10,v11,%[out])
        STORE64inc(v20,v21,v22,v23,%[out])
        STORE64inc(v16,v17,v18,v19,%[out])
        "                                   \n\t"
        " 5:                                \n\t" // End.
        "                                   \n\t"
    
        : // Output operands
          [mat]     "+r"(mat),
          [in]      "+r"(in),
          [out]     "+r"(out),
          [k]       "+r"(k)

        : // Input operands
          [k_odd]   "r"(k_odd),
          [pattern] "r"(pattern)

        : // Clobbered registers
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30"
    );
}

void microkernel_double_armv8a::
zm2l( const double *fac,
      const double *in,
            double *out,
      size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                   \n\t"
        " ldp q28, q29, [%[fac]], #32       \n\t" // First 4 values from faculty matrix
        "                                   \n\t"
        "                                   \n\t"  // Init result registers with zeros
        " eor v0.16b , v0.16b , v0.16b      \n\t"  // Row 0 of result
        " eor v1.16b , v1.16b , v1.16b      \n\t"  
        " eor v2.16b , v2.16b , v2.16b      \n\t"  
        " eor v3.16b , v3.16b , v3.16b      \n\t"  
        " eor v4.16b , v4.16b , v4.16b      \n\t"  // Row 1 of result
        " eor v5.16b , v5.16b , v5.16b      \n\t" 
        " eor v6.16b , v6.16b , v6.16b      \n\t"
        " eor v7.16b , v7.16b , v7.16b      \n\t"
        " eor v8.16b , v8.16b , v8.16b      \n\t"  // Row 2 of result
        " eor v9.16b , v9.16b , v9.16b      \n\t"
        " eor v10.16b, v10.16b, v10.16b     \n\t"
        " eor v11.16b, v11.16b, v11.16b     \n\t"
        " eor v12.16b, v12.16b, v12.16b     \n\t"  // Row 3 of result
        " eor v13.16b, v13.16b, v13.16b     \n\t"
        " eor v14.16b, v14.16b, v14.16b     \n\t"
        " eor v15.16b, v15.16b, v15.16b     \n\t"
        " eor v16.16b, v16.16b, v16.16b     \n\t"  // Row 4 of result
        " eor v17.16b, v17.16b, v17.16b     \n\t"
        " eor v18.16b, v18.16b, v18.16b     \n\t"
        " eor v19.16b, v19.16b, v19.16b     \n\t"
        " eor v20.16b, v20.16b, v20.16b     \n\t"  // Row 5 of result
        " eor v21.16b, v21.16b, v21.16b     \n\t"
        " eor v22.16b, v22.16b, v22.16b     \n\t"
        " eor v23.16b, v23.16b, v23.16b     \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ...
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        " ldr q30, [%[fac]], #8             \n\t" // Load last two faculties.
        "                                   \n\t" 
        DFMLA(v0,v24,v28,0)
        DFMLA(v1,v25,v28,0)
        DFMLA(v2,v26,v28,0)
        DFMLA(v3,v27,v28,0)
        "                                   \n\t"
        DFMLA(v4,v24,v28,1)
        DFMLA(v5,v25,v28,1)
        DFMLA(v6,v26,v28,1)
        DFMLA(v7,v27,v28,1)
        "                                   \n\t"
        DFMLA(v8,v24,v29,0)
        DFMLA(v9,v25,v29,0)
        DFMLA(v10,v26,v29,0)
        DFMLA(v11,v27,v29,0)
        "                                   \n\t"
        DFMLA(v12,v24,v29,1)
        DFMLA(v13,v25,v29,1)
        DFMLA(v14,v26,v29,1)
        DFMLA(v15,v27,v29,1)
        "                                   \n\t"
        DFMLA(v16,v24,v30,0)
        DFMLA(v17,v25,v30,0)
        DFMLA(v18,v26,v30,0)
        DFMLA(v19,v27,v30,0)
        "                                   \n\t"
        DFMLA(v20,v24,v30,1)
        DFMLA(v21,v25,v30,1)
        DFMLA(v22,v26,v30,1)
        DFMLA(v23,v27,v30,1)
        "                                   \n\t"
        " mov v28.d[0], v28.d[1]            \n\t" // "Shift" faculty regiesters.
        " mov v28.d[1], v29.d[0]            \n\t" 
        " mov v29.d[0], v29.d[1]            \n\t" 
        " mov v29.d[1], v30.d[0]            \n\t" 
        "                                   \n\t"
        " subs %[k], %[k], #1               \n\t"
        " b.ne 1b                           \n\t" // while ( --k );
        "                                   \n\t" 
        "                                   \n\t" 
        " cbz %w[pattern], 2f               \n\t" // if ( pattern )
        DFNEG(v0,v0)
        DFNEG(v1,v1)
        DFNEG(v2,v2)
        DFNEG(v3,v3)
        DFNEG(v8,v8)
        DFNEG(v9,v9)
        DFNEG(v10,v10)
        DFNEG(v11,v11)
        DFNEG(v16,v16)
        DFNEG(v17,v17)
        DFNEG(v18,v18)
        DFNEG(v19,v19)
        " b 3f                              \n\t" 
        " 2:                                \n\t" // else:
        DFNEG(v4,v4)
        DFNEG(v5,v5)
        DFNEG(v6,v6)
        DFNEG(v7,v7)
        DFNEG(v12,v12)
        DFNEG(v13,v13)
        DFNEG(v14,v14)
        DFNEG(v15,v15)
        DFNEG(v20,v20)
        DFNEG(v21,v21)
        DFNEG(v22,v22)
        DFNEG(v23,v23)
        "                                   \n\t" 
        " 3:                                \n\t" // Store result.
        STORE64inc(v0,v1,v2,v3,%[out])
        STORE64inc(v4,v5,v6,v7,%[out])
        STORE64inc(v8,v9,v10,v11,%[out])
        STORE64inc(v12,v13,v14,v15,%[out])
        STORE64inc(v16,v17,v18,v19,%[out])
        STORE64inc(v20,v21,v22,v23,%[out])

        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [out]     "+r"(out),
          [k]       "+r"(k)

        : // Input operands
          [pattern] "r"(pattern)

        : // Clobbered registers
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30"
    ); 
}

void microkernel_double_armv8a::
zm2m( const double *fac,
      const double *in,
            double *out,
            size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                   \n\t"
        " ldp q28, q29, [%[fac]], #32       \n\t"  // First 4 values from faculty matrix
        "                                   \n\t"
        "                                   \n\t"  // Init result registers with zeros
        " eor v0.16b , v0.16b , v0.16b      \n\t"  // Row 5 of result
        " eor v1.16b , v1.16b , v1.16b      \n\t"  
        " eor v2.16b , v2.16b , v2.16b      \n\t"  
        " eor v3.16b , v3.16b , v3.16b      \n\t"  
        " eor v4.16b , v4.16b , v4.16b      \n\t"  // Row 4 of result
        " eor v5.16b , v5.16b , v5.16b      \n\t" 
        " eor v6.16b , v6.16b , v6.16b      \n\t"
        " eor v7.16b , v7.16b , v7.16b      \n\t"
        " eor v8.16b , v8.16b , v8.16b      \n\t"  // Row 3 of result
        " eor v9.16b , v9.16b , v9.16b      \n\t"
        " eor v10.16b, v10.16b, v10.16b     \n\t"
        " eor v11.16b, v11.16b, v11.16b     \n\t"
        " eor v12.16b, v12.16b, v12.16b     \n\t"  // Row 2 of result
        " eor v13.16b, v13.16b, v13.16b     \n\t"
        " eor v14.16b, v14.16b, v14.16b     \n\t"
        " eor v15.16b, v15.16b, v15.16b     \n\t"
        " eor v16.16b, v16.16b, v16.16b     \n\t"  // Row 1 of result
        " eor v17.16b, v17.16b, v17.16b     \n\t"
        " eor v18.16b, v18.16b, v18.16b     \n\t"
        " eor v19.16b, v19.16b, v19.16b     \n\t"
        " eor v20.16b, v20.16b, v20.16b     \n\t"  // Row 0 of result
        " eor v21.16b, v21.16b, v21.16b     \n\t"
        " eor v22.16b, v22.16b, v22.16b     \n\t"
        " eor v23.16b, v23.16b, v23.16b     \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ...
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])        // Load next row if input
        " ldr q30, [%[fac]]                 \n\t" // Load last two faculties
        "                                   \n\t" 
        DFMLA(v0,v24,v28,0)
        DFMLA(v1,v25,v28,0)
        DFMLA(v2,v26,v28,0)
        DFMLA(v3,v27,v28,0)
        "                                   \n\t"
        DFMLA(v4,v24,v28,1)
        DFMLA(v5,v25,v28,1)
        DFMLA(v6,v26,v28,1)
        DFMLA(v7,v27,v28,1)
        "                                   \n\t"
        DFMLA(v8,v24,v29,0)
        DFMLA(v9,v25,v29,0)
        DFMLA(v10,v26,v29,0)
        DFMLA(v11,v27,v29,0)
        "                                   \n\t"
        DFMLA(v12,v24,v29,1)
        DFMLA(v13,v25,v29,1)
        DFMLA(v14,v26,v29,1)
        DFMLA(v15,v27,v29,1)
        "                                   \n\t"
        DFMLA(v16,v24,v30,0)
        DFMLA(v17,v25,v30,0)
        DFMLA(v18,v26,v30,0)
        DFMLA(v19,v27,v30,0)
        "                                   \n\t"
        DFMLA(v20,v24,v30,1)
        DFMLA(v21,v25,v30,1)
        DFMLA(v22,v26,v30,1)
        DFMLA(v23,v27,v30,1)
        "                                   \n\t"
        " mov v28.d[0], v28.d[1]            \n\t" // "Shift" faculty regiesters.
        " mov v28.d[1], v29.d[0]            \n\t" 
        " mov v29.d[0], v29.d[1]            \n\t" 
        " mov v29.d[1], v30.d[0]            \n\t"
        "                                   \n\t"
        " add %[fac], %[fac], #8            \n\t"              
        " subs %[k], %[k], #1               \n\t"
        " b.ne 1b                           \n\t" // while ( --k );
        "                                   \n\t" 
        "                                   \n\t" // Store result.
        STORE64inc(v20,v21,v22,v23,%[out])
        STORE64inc(v16,v17,v18,v19,%[out])
        STORE64inc(v12,v13,v14,v15,%[out])
        STORE64inc(v8,v9,v10,v11,%[out])
        STORE64inc(v4,v5,v6,v7,%[out])
        STORE64inc(v0,v1,v2,v3,%[out])

        : // Output operands
          [fac]     "+r"(fac),
          [in]      "+r"(in),
          [out]     "+r"(out),
          [k]       "+r"(k)

        : // Input operands

        : // Clobbered registers
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v8" , "v9" , "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30"
    ); 
}


void microkernel_double_armv8a::
swap2trans_buf( const double  *real_in, 
                const double  *imag_in,
                      double **real_out, 
                      double **imag_out,
                      size_t   n ) const noexcept
{
    size_t m = 0;
    __asm__ volatile
    (
        "                                     \n\t" 
        " .align 5                            \n\t"
        " 1:                                  \n\t" 
        "ldr x5, [ %[real_out], %[m], LSL#3 ] \n\t" // x5 = real_out[m]
        "ldr x6, [ %[imag_out], %[m], LSL#3 ] \n\t" // x6 = imag_out[m]
        "sub x7, %[n], %[m]                   \n\t" // x7 = n - m
        "add x5, x5, x7, LSL#6                \n\t" // x5 = real_out[m] + (n-m)*cols
        "add x6, x6, x7, LSL#6                \n\t" // x6 = imag_out[m] + (n-m)*cols

        LOAD64inc(v0,v1,v2,v3,%[real_in])
        LOAD64inc(v4,v5,v6,v7,%[imag_in])
        STORE64(v0,v1,v2,v3,x5)
        STORE64(v4,v5,v6,v7,x6)

        "cmp %[m], %[n]                       \n\t"
        "add %[m], %[m], #1                   \n\t"
        "b.ne 1b                              \n\t"

        : // Output operands
          [real_in]  "+r"(real_in),
          [imag_in]  "+r"(imag_in),
          [m]        "+r"(m)
    
        : // Input operands
          [real_out] "r"(real_out),
          [imag_out] "r"(imag_out),
          [n]        "r"(n)

        : // Clobbered registers
        "x5", "x6", "x7",
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
    );
}

void microkernel_double_armv8a::
trans2swap_buf( const double *const *const  real_in, 
                const double *const *const  imag_in,
                      double               *real_out, 
                      double               *imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    size_t m = 0;
    __asm__ volatile
    (
        "                                     \n\t"
        " eor v16.16b, v16.16b, v16.16b       \n\t"
        " eor v17.16b, v17.16b, v17.16b       \n\t"
        " eor v18.16b, v18.16b, v18.16b       \n\t"
        " eor v19.16b, v19.16b, v19.16b       \n\t"
        "                                     \n\t"
        " add x10, %[n], #1                   \n\t" // x10 = n+1
        " cmp x10, %[Pmax]                    \n\t"
        " b.ls 1f                             \n\t"
        " mov x10, %[Pmax]                    \n\t" // x10 = min(Pmax,n+1)
        "                                     \n\t"
        " 1:                                  \n\t"
        " cmp x10, %[m]                       \n\t" // while ( m < min(Pmax,n+1) )
        " b.ls 3f                             \n\t"
        "                                     \n\t"
        " .align 5                            \n\t"
        " 2:                                  \n\t"
        " ldr x7, [ %[real_in], %[m], LSL#3 ] \n\t" // x7 = real_in[m]
        " ldr x8, [ %[imag_in], %[m], LSL#3 ] \n\t" // x8 = imag_in[m]
        " sub x9, %[n], %[m]                  \n\t" // x9 = n - m
        " add x7, x7, x9, LSL#6               \n\t" // x7 = real_in[m] + (n-m)*cols
        " add x8, x8, x9, LSL#6               \n\t" // x8 = imag_in[m] + (n-m)*cols
        "                                     \n\t"
        LOAD64(v0,v1,v2,v3,x7)
        LOAD64(v4,v5,v6,v7,x8)
        STORE64inc(v0,v1,v2,v3,%[real_out])
        STORE64inc(v4,v5,v6,v7,%[imag_out])
        " add %[m], %[m], #1                  \n\t"
        " cmp %[m], x10                       \n\t"
        " b.lo 2b                             \n\t"
        "                                     \n\t"
        " 3:                                  \n\t" 
        " cmp %[n], %[m]                      \n\t" // while ( m <= n )
        " b.lo 5f                             \n\t"
        "                                     \n\t" 
        " sub %[m], %[n], %[m]                \n\t"
        " add %[m], %[m], 1                   \n\t" // Number of iterations: n-m+1
        "                                     \n\t"
        " .align 5                            \n\t"
        " 4:                                  \n\t" 
        STORE64inc(v16,v17,v18,v19,%[real_out])     // Store zeros.
        STORE64inc(v16,v17,v18,v19,%[imag_out])
        " subs %[m], %[m], #1                 \n\t"
        " b.ne 4b                             \n\t"
        "                                     \n\t"
        " 5:                                  \n\t"
        : // Output operands
          [real_out] "+r"(real_out),
          [imag_out] "+r"(imag_out),
          [m]        "+r"(m)
    
        : // Input operands
          [real_in]  "r"(real_in),
          [imag_in]  "r"(imag_in),
          [n]        "r"(n),
          [Pmax]     "r"(Pmax)

        : // Clobbered registers
        "x7" , "x8" , "x9" , "x10",
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v16", "v17", "v18", "v19"
    );
}

void microkernel_double_armv8a::
solid2buf( const double *const *p_solids,
           const double *const  zeros,
           const size_t        *P,
                 double        *real_out,
                 double        *imag_out,
                 size_t         n ) const noexcept
{
    const double* re[ 8 ] =
    {
        ( n < P[0] ) ? ( p_solids[0] + (n  )*(n+1) ) : zeros,
        ( n < P[1] ) ? ( p_solids[1] + (n  )*(n+1) ) : zeros,
        ( n < P[2] ) ? ( p_solids[2] + (n  )*(n+1) ) : zeros,
        ( n < P[3] ) ? ( p_solids[3] + (n  )*(n+1) ) : zeros,
        ( n < P[4] ) ? ( p_solids[4] + (n  )*(n+1) ) : zeros,
        ( n < P[5] ) ? ( p_solids[5] + (n  )*(n+1) ) : zeros,
        ( n < P[6] ) ? ( p_solids[6] + (n  )*(n+1) ) : zeros,
        ( n < P[7] ) ? ( p_solids[7] + (n  )*(n+1) ) : zeros
    };

    size_t m = n+1; // Number of elements we actually need to copy.

    __asm__ volatile
    (
        "                                     \n\t"
        " add x20, %[re0], %[m], lsl #3       \n\t" // Addresses of imaginary parts.
        " add x21, %[re1], %[m], lsl #3       \n\t"
        " add x22, %[re2], %[m], lsl #3       \n\t"
        " add x23, %[re3], %[m], lsl #3       \n\t"
        " add x24, %[re4], %[m], lsl #3       \n\t"
        " add x25, %[re5], %[m], lsl #3       \n\t"
        " add x26, %[re6], %[m], lsl #3       \n\t"
        " add x27, %[re7], %[m], lsl #3       \n\t"
        "                                     \n\t"
        " cmp %[m], #4                        \n\t" // The following loop copies 4 elements per iteration
        " b.lo 3f                             \n\t" // Skip if we have less four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 4:                                  \n\t"
        " ldp q0, q16, [%[re0]], #32          \n\t"
        " ldp q1, q17, [%[re1]], #32          \n\t"
        " ldp q2, q18, [%[re2]], #32          \n\t"
        " ldp q3, q19, [%[re3]], #32          \n\t"
        " ldp q4, q20, [%[re4]], #32          \n\t"
        " ldp q5, q21, [%[re5]], #32          \n\t"
        " ldp q6, q22, [%[re6]], #32          \n\t"
        " ldp q7, q23, [%[re7]], #32          \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[real_out])
        DST4inc(v4,v5,v6,v7,0,%[real_out])
        DST4inc(v0,v1,v2,v3,1,%[real_out])
        DST4inc(v4,v5,v6,v7,1,%[real_out])
        DST4inc(v16,v17,v18,v19,0,%[real_out])
        DST4inc(v20,v21,v22,v23,0,%[real_out])
        DST4inc(v16,v17,v18,v19,1,%[real_out])
        DST4inc(v20,v21,v22,v23,1,%[real_out])
        "                                     \n\t"
        " ldp q0, q16, [x20], #32             \n\t"
        " ldp q1, q17, [x21], #32             \n\t"
        " ldp q2, q18, [x22], #32             \n\t"
        " ldp q3, q19, [x23], #32             \n\t"
        " ldp q4, q20, [x24], #32             \n\t"
        " ldp q5, q21, [x25], #32             \n\t"
        " ldp q6, q22, [x26], #32             \n\t"
        " ldp q7, q23, [x27], #32             \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[imag_out])
        DST4inc(v4,v5,v6,v7,0,%[imag_out])
        DST4inc(v0,v1,v2,v3,1,%[imag_out])
        DST4inc(v4,v5,v6,v7,1,%[imag_out])
        DST4inc(v16,v17,v18,v19,0,%[imag_out])
        DST4inc(v20,v21,v22,v23,0,%[imag_out])
        DST4inc(v16,v17,v18,v19,1,%[imag_out])
        DST4inc(v20,v21,v22,v23,1,%[imag_out])
        "                                     \n\t"
        " sub %[m], %[m], #4                  \n\t"
        " cmp %[m], #4                        \n\t"
        " b.hs 4b                             \n\t"
        "                                     \n\t"
        " 3:                                  \n\t"
        " cmp %[m], #2                        \n\t"
        " b.lo 1f                             \n\t"
        " ldr q0, [%[re0]], #16               \n\t"
        " ldr q1, [%[re1]], #16               \n\t"
        " ldr q2, [%[re2]], #16               \n\t"
        " ldr q3, [%[re3]], #16               \n\t"
        " ldr q4, [%[re4]], #16               \n\t"
        " ldr q5, [%[re5]], #16               \n\t"
        " ldr q6, [%[re6]], #16               \n\t"
        " ldr q7, [%[re7]], #16               \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[real_out])
        DST4inc(v4,v5,v6,v7,0,%[real_out])
        DST4inc(v0,v1,v2,v3,1,%[real_out])
        DST4inc(v4,v5,v6,v7,1,%[real_out])
        "                                     \n\t"
        " ldr q0, [x20], #16                  \n\t"
        " ldr q1, [x21], #16                  \n\t"
        " ldr q2, [x22], #16                  \n\t"
        " ldr q3, [x23], #16                  \n\t"
        " ldr q4, [x24], #16                  \n\t"
        " ldr q5, [x25], #16                  \n\t"
        " ldr q6, [x26], #16                  \n\t"
        " ldr q7, [x27], #16                  \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[imag_out])
        DST4inc(v4,v5,v6,v7,0,%[imag_out])
        DST4inc(v0,v1,v2,v3,1,%[imag_out])
        DST4inc(v4,v5,v6,v7,1,%[imag_out])
        "                                     \n\t"
        " sub %[m], %[m], #2                  \n\t" 
        "                                     \n\t"
        " 1:                                  \n\t"
        " cbz %[m], 0f                        \n\t"
        " ldr d0, [%[re0]]                    \n\t"
        " ldr d1, [%[re1]]                    \n\t"
        " ldr d2, [%[re2]]                    \n\t"
        " ldr d3, [%[re3]]                    \n\t"
        " ldr d4, [%[re4]]                    \n\t"
        " ldr d5, [%[re5]]                    \n\t"
        " ldr d6, [%[re6]]                    \n\t"
        " ldr d7, [%[re7]]                    \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[real_out])
        DST4inc(v4,v5,v6,v7,0,%[real_out])
        "                                     \n\t"
        " ldr d0, [x20]                       \n\t"
        " ldr d1, [x21]                       \n\t"
        " ldr d2, [x22]                       \n\t"
        " ldr d3, [x23]                       \n\t"
        " ldr d4, [x24]                       \n\t"
        " ldr d5, [x25]                       \n\t"
        " ldr d6, [x26]                       \n\t"
        " ldr d7, [x27]                       \n\t"
        "                                     \n\t"
        DST4inc(v0,v1,v2,v3,0,%[imag_out])
        DST4inc(v4,v5,v6,v7,0,%[imag_out])
        "                                     \n\t"
        " 0:                                  \n\t"

        : // Output
            [m]        "+r"(m),
            [real_out] "+r"(real_out),
            [imag_out] "+r"(imag_out),
            [re0]      "+r"(re[0]),
            [re1]      "+r"(re[1]),
            [re2]      "+r"(re[2]),
            [re3]      "+r"(re[3]),
            [re4]      "+r"(re[4]),
            [re5]      "+r"(re[5]),
            [re6]      "+r"(re[6]),
            [re7]      "+r"(re[7])

        : // Input

        : // Clobbered registers
        "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
    );
}

void microkernel_double_armv8a::
buf2solid( const double  *real_in, 
           const double  *imag_in, 
                 double **p_solids,
                 double  *trash, 
           const size_t  *P,
                 size_t   n ) const noexcept
{
    double* re[ 8 ] =
    {
        ( n < P[0] ) ? ( p_solids[0] + (n  )*(n+1) ) : trash,
        ( n < P[1] ) ? ( p_solids[1] + (n  )*(n+1) ) : trash,
        ( n < P[2] ) ? ( p_solids[2] + (n  )*(n+1) ) : trash,
        ( n < P[3] ) ? ( p_solids[3] + (n  )*(n+1) ) : trash,
        ( n < P[4] ) ? ( p_solids[4] + (n  )*(n+1) ) : trash,
        ( n < P[5] ) ? ( p_solids[5] + (n  )*(n+1) ) : trash,
        ( n < P[6] ) ? ( p_solids[6] + (n  )*(n+1) ) : trash,
        ( n < P[7] ) ? ( p_solids[7] + (n  )*(n+1) ) : trash
    };

    size_t m = n+1; // Number of elements we actually need to copy.

    __asm__ volatile
    (
        "                                     \n\t"
        " add x20, %[re0], %[m], lsl #3       \n\t" // Addresses of imaginary parts.
        " add x21, %[re1], %[m], lsl #3       \n\t"
        " add x22, %[re2], %[m], lsl #3       \n\t"
        " add x23, %[re3], %[m], lsl #3       \n\t"
        " add x24, %[re4], %[m], lsl #3       \n\t"
        " add x25, %[re5], %[m], lsl #3       \n\t"
        " add x26, %[re6], %[m], lsl #3       \n\t"
        " add x27, %[re7], %[m], lsl #3       \n\t"
        "                                     \n\t"
        " cmp %[m], #4                        \n\t" // The following loop copies 4 elements per iteration
        " b.lo 3f                             \n\t" // Skip if we have less four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 4:                                  \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[real_in])
        DLD4inc(v4,v5,v6,v7,0,%[real_in])
        DLD4inc(v0,v1,v2,v3,1,%[real_in])
        DLD4inc(v4,v5,v6,v7,1,%[real_in])
        DLD4inc(v16,v17,v18,v19,0,%[real_in])
        DLD4inc(v20,v21,v22,v23,0,%[real_in])
        DLD4inc(v16,v17,v18,v19,1,%[real_in])
        DLD4inc(v20,v21,v22,v23,1,%[real_in])
        "                                     \n\t"
        " ldp q24, q25, [%[re0]]              \n\t"
        DFADD(v24,v24,v0)
        DFADD(v25,v25,v16)
        " stp q24, q25, [%[re0]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re1]]              \n\t"
        DFADD(v24,v24,v1)
        DFADD(v25,v25,v17)
        " stp q24, q25, [%[re1]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re2]]              \n\t"
        DFADD(v24,v24,v2)
        DFADD(v25,v25,v18)
        " stp q24, q25, [%[re2]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re3]]              \n\t"
        DFADD(v24,v24,v3)
        DFADD(v25,v25,v19)
        " stp q24, q25, [%[re3]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re4]]              \n\t"
        DFADD(v24,v24,v4)
        DFADD(v25,v25,v20)
        " stp q24, q25, [%[re4]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re5]]              \n\t"
        DFADD(v24,v24,v5)
        DFADD(v25,v25,v21)
        " stp q24, q25, [%[re5]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re6]]              \n\t"
        DFADD(v24,v24,v6)
        DFADD(v25,v25,v22)
        " stp q24, q25, [%[re6]], #32         \n\t"
        "                                     \n\t"
        " ldp q24, q25, [%[re7]]              \n\t"
        DFADD(v24,v24,v7)
        DFADD(v25,v25,v23)
        " stp q24, q25, [%[re7]], #32         \n\t"
        "                                     \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[imag_in])
        DLD4inc(v4,v5,v6,v7,0,%[imag_in])
        DLD4inc(v0,v1,v2,v3,1,%[imag_in])
        DLD4inc(v4,v5,v6,v7,1,%[imag_in])
        DLD4inc(v16,v17,v18,v19,0,%[imag_in])
        DLD4inc(v20,v21,v22,v23,0,%[imag_in])
        DLD4inc(v16,v17,v18,v19,1,%[imag_in])
        DLD4inc(v20,v21,v22,v23,1,%[imag_in])
        "                                     \n\t"
        " ldp q24, q25, [x20]                 \n\t"
        DFADD(v24,v24,v0)
        DFADD(v25,v25,v16)
        " stp q24, q25, [x20], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x21]                 \n\t"
        DFADD(v24,v24,v1)
        DFADD(v25,v25,v17)
        " stp q24, q25, [x21], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x22]                 \n\t"
        DFADD(v24,v24,v2)
        DFADD(v25,v25,v18)
        " stp q24, q25, [x22], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x23]                 \n\t"
        DFADD(v24,v24,v3)
        DFADD(v25,v25,v19)
        " stp q24, q25, [x23], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x24]                 \n\t"
        DFADD(v24,v24,v4)
        DFADD(v25,v25,v20)
        " stp q24, q25, [x24], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x25]                 \n\t"
        DFADD(v24,v24,v5)
        DFADD(v25,v25,v21)
        " stp q24, q25, [x25], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x26]                 \n\t"
        DFADD(v24,v24,v6)
        DFADD(v25,v25,v22)
        " stp q24, q25, [x26], #32            \n\t"
        "                                     \n\t"
        " ldp q24, q25, [x27]                 \n\t"
        DFADD(v24,v24,v7)
        DFADD(v25,v25,v23)
        " stp q24, q25, [x27], #32            \n\t"
        "                                     \n\t"
        " sub %[m], %[m], #4                  \n\t"
        " cmp %[m], #4                        \n\t"
        " b.hs 4b                             \n\t"
        "                                     \n\t"
        " 3:                                  \n\t"
        " cmp %[m], #2                        \n\t"
        " b.lo 1f                             \n\t"
        "                                     \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[real_in])
        DLD4inc(v4,v5,v6,v7,0,%[real_in])
        DLD4inc(v0,v1,v2,v3,1,%[real_in])
        DLD4inc(v4,v5,v6,v7,1,%[real_in])
        "                                     \n\t"
        " ldr q24, [%[re0]]                   \n\t"
        DFADD(v24,v24,v0)
        " str q24, [%[re0]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re1]]                   \n\t"
        DFADD(v24,v24,v1)
        " str q24, [%[re1]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re2]]                   \n\t"
        DFADD(v24,v24,v2)
        " str q24, [%[re2]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re3]]                   \n\t"
        DFADD(v24,v24,v3)
        " str q24, [%[re3]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re4]]                   \n\t"
        DFADD(v24,v24,v4)
        " str q24, [%[re4]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re5]]                   \n\t"
        DFADD(v24,v24,v5)
        " str q24, [%[re5]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re6]]                   \n\t"
        DFADD(v24,v24,v6)
        " str q24, [%[re6]], #16              \n\t"
        "                                     \n\t"
        " ldr q24, [%[re7]]                   \n\t"
        DFADD(v24,v24,v7)
        " str q24, [%[re7]], #16              \n\t"
        "                                     \n\t"
        "                                     \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[imag_in])
        DLD4inc(v4,v5,v6,v7,0,%[imag_in])
        DLD4inc(v0,v1,v2,v3,1,%[imag_in])
        DLD4inc(v4,v5,v6,v7,1,%[imag_in])
        "                                     \n\t"
        " ldr q24, [x20]                      \n\t"
        DFADD(v24,v24,v0)
        " str q24, [x20], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x21]                      \n\t"
        DFADD(v24,v24,v1)
        " str q24, [x21], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x22]                      \n\t"
        DFADD(v24,v24,v2)
        " str q24, [x22], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x23]                      \n\t"
        DFADD(v24,v24,v3)
        " str q24, [x23], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x24]                      \n\t"
        DFADD(v24,v24,v4)
        " str q24, [x24], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x25]                      \n\t"
        DFADD(v24,v24,v5)
        " str q24, [x25], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x26]                      \n\t"
        DFADD(v24,v24,v6)
        " str q24, [x26], #16                 \n\t"
        "                                     \n\t"
        " ldr q24, [x27]                      \n\t"
        DFADD(v24,v24,v7)
        " str q24, [x27], #16                 \n\t"
        "                                     \n\t"
        " sub %[m], %[m], #2                  \n\t"
        "                                     \n\t"
        " 1:                                  \n\t"
        " cbz %[m], 0f                        \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[real_in])
        DLD4inc(v4,v5,v6,v7,0,%[real_in])
        "                                     \n\t"
        " ldr d24, [%[re0]]                   \n\t"
        " fadd d24, d24, d0                   \n\t"
        " str d24, [%[re0]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re1]]                   \n\t"
        " fadd d24, d24, d1                   \n\t"
        " str d24, [%[re1]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re2]]                   \n\t"
        " fadd d24, d24, d2                   \n\t"
        " str d24, [%[re2]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re3]]                   \n\t"
        " fadd d24, d24, d3                   \n\t"
        " str d24, [%[re3]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re4]]                   \n\t"
        " fadd d24, d24, d4                   \n\t"
        " str d24, [%[re4]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re5]]                   \n\t"
        " fadd d24, d24, d5                   \n\t"
        " str d24, [%[re5]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re6]]                   \n\t"
        " fadd d24, d24, d6                   \n\t"
        " str d24, [%[re6]]                   \n\t"
        "                                     \n\t"
        " ldr d24, [%[re7]]                   \n\t"
        " fadd d24, d24, d7                   \n\t"
        " str d24, [%[re7]]                   \n\t"
        "                                     \n\t"
        DLD4inc(v0,v1,v2,v3,0,%[imag_in])
        DLD4inc(v4,v5,v6,v7,0,%[imag_in])
        "                                     \n\t"
        " ldr d24, [x20]                      \n\t"
        " fadd d24, d24, d0                   \n\t"
        " str d24, [x20]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x21]                      \n\t"
        " fadd d24, d24, d1                   \n\t"
        " str d24, [x21]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x22]                      \n\t"
        " fadd d24, d24, d2                   \n\t"
        " str d24, [x22]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x23]                      \n\t"
        " fadd d24, d24, d3                   \n\t"
        " str d24, [x23]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x24]                      \n\t"
        " fadd d24, d24, d4                   \n\t"
        " str d24, [x24]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x25]                      \n\t"
        " fadd d24, d24, d5                   \n\t"
        " str d24, [x25]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x26]                      \n\t"
        " fadd d24, d24, d6                   \n\t"
        " str d24, [x26]                      \n\t"
        "                                     \n\t"
        " ldr d24, [x27]                      \n\t"
        " fadd d24, d24, d7                   \n\t"
        " str d24, [x27]                      \n\t"
        "                                     \n\t"
        " 0:                                  \n\t"

        : // Output
            [m]        "+r"(m),
            [real_in]  "+r"(real_in),
            [imag_in]  "+r"(imag_in),
            [re0]      "+r"(re[0]),
            [re1]      "+r"(re[1]),
            [re2]      "+r"(re[2]),
            [re3]      "+r"(re[3]),
            [re4]      "+r"(re[4]),
            [re5]      "+r"(re[5]),
            [re6]      "+r"(re[6]),
            [re7]      "+r"(re[7])

        : // Input

        : // Clobbered registers
        "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
        "v0" , "v1" , "v2" , "v3" , "v4" , "v5" , "v6" , "v7" ,
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25"
    );
}

}

#endif

