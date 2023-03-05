/*
 * Copyright (C) 2022, 2023 Matthias Kirchhart
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
#define FFMLA(a,b,c,n) " fmla.4s "#a","#b","#c"["#n"]         \n\t"
#define DFMLAV(a,b,c)  " fmla.2d "#a","#b","#c"               \n\t"
#define FFMLAV(a,b,c)  " fmla.4s "#a","#b","#c"               \n\t"
#define DFMLSV(a,b,c)  " fmls.2d "#a","#b","#c"               \n\t"
#define FFMLSV(a,b,c)  " fmls.4s "#a","#b","#c"               \n\t"
#define DFMUL(a,b,c)   " fmul.2d "#a","#b","#c"               \n\t"
#define FFMUL(a,b,c)   " fmul.4s "#a","#b","#c"               \n\t"
#define DFADD(a,b,c)   " fadd.2d "#a","#b","#c"               \n\t"
#define FFADD(a,b,c)   " fadd.4s "#a","#b","#c"               \n\t"
#define DFDIV(a,b,c)   " fdiv.2d "#a","#b","#c"               \n\t"
#define FFDIV(a,b,c)   " fdiv.4s "#a","#b","#c"               \n\t"
#define DFSQRT(a,b)    " fsqrt.2d "#a","#b"                   \n\t"
#define FFSQRT(a,b)    " fsqrt.4s "#a","#b"                   \n\t"
#define DFCMEQZ(a,b)   " fcmeq.2d "#a","#b", #0.0             \n\t"
#define FFCMEQZ(a,b)   " fcmeq.4s "#a","#b", #0.0             \n\t"
#define DFNEG(a,b)     " fneg.2d  "#a","#b"                   \n\t"
#define FFNEG(a,b)     " fneg.4s  "#a","#b"                   \n\t"
#define LOAD64(a,b,c,d,e)     " ld1.16b { "#a", "#b", "#c", "#d" }, ["#e"]      \n\t"
#define LOAD64inc(a,b,c,d,e)  " ld1.16b { "#a", "#b", "#c", "#d" }, ["#e"], #64 \n\t"
#define LOAD48inc(a,b,c,d)    " ld1.16b { "#a", "#b", "#c" }, ["#d"], #48       \n\t"
#define LOAD48(a,b,c,d)       " ld1.16b { "#a", "#b", "#c" }, ["#d"]            \n\t"
#define STORE64(a,b,c,d,e)    " st1.16b { "#a", "#b", "#c", "#d" }, ["#e"]      \n\t"
#define STORE64inc(a,b,c,d,e) " st1.16b { "#a", "#b", "#c", "#d" }, ["#e"], #64 \n\t"
#define DST4inc(a,b,c,d,e,f)  " st4.d   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #32 \n\t"
#define FST4inc(a,b,c,d,e,f)  " st4.s   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #16 \n\t"
#define DLD4inc(a,b,c,d,e,f)  " ld4.d   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #32 \n\t"
#define FLD4inc(a,b,c,d,e,f)  " ld4.s   { "#a", "#b", "#c", "#d" }["#e"], ["#f"], #16 \n\t"
#else
#define DFMLA(a,b,c,n) " fmla    "#a".2d,"#b".2d,"#c".d["#n"] \n\t"
#define FFMLA(a,b,c,n) " fmla    "#a".4s,"#b".4s,"#c".s["#n"] \n\t"
#define DFMLAV(a,b,c)  " fmla    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define FFMLAV(a,b,c)  " fmla    "#a".4s,"#b".4s,"#c".4s      \n\t"
#define DFMLSV(a,b,c)  " fmls    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define FFMLSV(a,b,c)  " fmls    "#a".4s,"#b".4s,"#c".4s      \n\t"
#define DFMUL(a,b,c)   " fmul    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define FFMUL(a,b,c)   " fmul    "#a".4s,"#b".4s,"#c".4s      \n\t"
#define DFADD(a,b,c)   " fadd    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define FFADD(a,b,c)   " fadd    "#a".4s,"#b".4s,"#c".4s      \n\t"
#define DFDIV(a,b,c)   " fdiv    "#a".2d,"#b".2d,"#c".2d      \n\t"
#define FFDIV(a,b,c)   " fdiv    "#a".4s,"#b".4s,"#c".4s      \n\t"
#define DFSQRT(a,b)    " fsqrt   "#a".2d,"#b".2d              \n\t"
#define FFSQRT(a,b)    " fsqrt   "#a".4s,"#b".4s              \n\t"
#define DFCMEQZ(a,b)   " fcmeq   "#a".2d,"#b".2d, #0.0        \n\t"
#define FFCMEQZ(a,b)   " fcmeq   "#a".4s,"#b".4s, #0.0        \n\t"
#define DFNEG(a,b)     " fneg    "#a".2d,"#b".2d              \n\t"
#define FFNEG(a,b)     " fneg    "#a".4s,"#b".4s              \n\t"
#define LOAD64(a,b,c,d,e)     " ld1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"]      \n\t"
#define LOAD64inc(a,b,c,d,e)  " ld1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"], #64 \n\t"
#define LOAD48inc(a,b,c,d)    " ld1 { "#a".16b, "#b".16b, "#c".16b }, ["#d"], #48           \n\t"
#define LOAD48(a,b,c,d)       " ld1 { "#a".16b, "#b".16b, "#c".16b }, ["#d"]                \n\t"
#define STORE64(a,b,c,d,e)    " st1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"]      \n\t"
#define STORE64inc(a,b,c,d,e) " st1 { "#a".16b, "#b".16b, "#c".16b, "#d".16b }, ["#e"], #64 \n\t"
#define DST4inc(a,b,c,d,e,f)  " st4 { "#a".d,   "#b".d,   "#c".d,   "#d".d  }["#e"], ["#f"], #32 \n\t"
#define FST4inc(a,b,c,d,e,f)  " st4 { "#a".s,   "#b".s,   "#c".s,   "#d".s  }["#e"], ["#f"], #16 \n\t"
#define DLD4inc(a,b,c,d,e,f)  " ld4 { "#a".d,   "#b".d,   "#c".d,   "#d".d  }["#e"], ["#f"], #32 \n\t"
#define FLD4inc(a,b,c,d,e,f)  " ld4 { "#a".s,   "#b".s,   "#c".s,   "#d".s  }["#e"], ["#f"], #16 \n\t"
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
        " cmp  %[k], #3                     \n\t"
        " b.lo 2f                           \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ... One iteration does three columns
        "                                   \n\t" // of the faculty matrix
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        LOAD64(v28,v29,v30,v31,%[fac])            // Load faculties for next three colums
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
        " add %[fac], %[fac], #24           \n\t"
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
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        DFMLA(v0,v24,v28,1)
        DFMLA(v1,v25,v28,1)
        DFMLA(v2,v26,v28,1)
        DFMLA(v3,v27,v28,1)
        "                                   \n\t"
        DFMLA(v4,v24,v29,0)
        DFMLA(v5,v25,v29,0)
        DFMLA(v6,v26,v29,0)
        DFMLA(v7,v27,v29,0)
        "                                   \n\t"
        DFMLA(v8,v24,v29,1)
        DFMLA(v9,v25,v29,1)
        DFMLA(v10,v26,v29,1)
        DFMLA(v11,v27,v29,1)
        "                                   \n\t"
        DFMLA(v12,v24,v30,0)
        DFMLA(v13,v25,v30,0)
        DFMLA(v14,v26,v30,0)
        DFMLA(v15,v27,v30,0)
        "                                   \n\t"
        DFMLA(v16,v24,v30,1)
        DFMLA(v17,v25,v30,1)
        DFMLA(v18,v26,v30,1)
        DFMLA(v19,v27,v30,1)
        "                                   \n\t"
        DFMLA(v20,v24,v31,0)
        DFMLA(v21,v25,v31,0)
        DFMLA(v22,v26,v31,0)
        DFMLA(v23,v27,v31,0)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        DFMLA(v0,v24,v29,0)
        DFMLA(v1,v25,v29,0)
        DFMLA(v2,v26,v29,0)
        DFMLA(v3,v27,v29,0)
        "                                   \n\t"
        DFMLA(v4,v24,v29,1)
        DFMLA(v5,v25,v29,1)
        DFMLA(v6,v26,v29,1)
        DFMLA(v7,v27,v29,1)
        "                                   \n\t"
        DFMLA(v8,v24,v30,0)
        DFMLA(v9,v25,v30,0)
        DFMLA(v10,v26,v30,0)
        DFMLA(v11,v27,v30,0)
        "                                   \n\t"
        DFMLA(v12,v24,v30,1)
        DFMLA(v13,v25,v30,1)
        DFMLA(v14,v26,v30,1)
        DFMLA(v15,v27,v30,1)
        "                                   \n\t"
        DFMLA(v16,v24,v31,0)
        DFMLA(v17,v25,v31,0)
        DFMLA(v18,v26,v31,0)
        DFMLA(v19,v27,v31,0)
        "                                   \n\t"
        DFMLA(v20,v24,v31,1)
        DFMLA(v21,v25,v31,1)
        DFMLA(v22,v26,v31,1)
        DFMLA(v23,v27,v31,1)
        "                                   \n\t"
        " sub  %[k], %[k], #3               \n\t"
        " cmp  %[k], #3                     \n\t"
        " b.hs 1b                           \n\t" // while ( --k );
        "                                   \n\t" 
        " 2:                                \n\t" 
        " cbz %[k], 4f                      \n\t" // Take care of the rest if there is some
        "                                   \n\t" 
        " 3:                                \n\t" 
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        LOAD48(v28,v29,v30,%[fac])            
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
        " add %[fac], %[fac], #8            \n\t"
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
        " subs %[k], %[k], 1                \n\t"
        " b.ne 3b                           \n\t"
        "                                   \n\t"
        " 4:                                \n\t"
        " cbz %w[pattern], 5f               \n\t" // if ( pattern )
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
        " b 6f                              \n\t" 
        " 5:                                \n\t" // else:
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
        " 6:                                \n\t" // Store result.
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
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
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
        " cmp  %[k], #3                     \n\t"
        " b.lo 2f                           \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ... One iteration does three columns
        "                                   \n\t" // of the faculty matrix
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        LOAD64(v28,v29,v30,v31,%[fac])            // Load faculties for next three colums
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
        " add %[fac], %[fac], #24           \n\t"
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
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        DFMLA(v0,v24,v28,1)
        DFMLA(v1,v25,v28,1)
        DFMLA(v2,v26,v28,1)
        DFMLA(v3,v27,v28,1)
        "                                   \n\t"
        DFMLA(v4,v24,v29,0)
        DFMLA(v5,v25,v29,0)
        DFMLA(v6,v26,v29,0)
        DFMLA(v7,v27,v29,0)
        "                                   \n\t"
        DFMLA(v8,v24,v29,1)
        DFMLA(v9,v25,v29,1)
        DFMLA(v10,v26,v29,1)
        DFMLA(v11,v27,v29,1)
        "                                   \n\t"
        DFMLA(v12,v24,v30,0)
        DFMLA(v13,v25,v30,0)
        DFMLA(v14,v26,v30,0)
        DFMLA(v15,v27,v30,0)
        "                                   \n\t"
        DFMLA(v16,v24,v30,1)
        DFMLA(v17,v25,v30,1)
        DFMLA(v18,v26,v30,1)
        DFMLA(v19,v27,v30,1)
        "                                   \n\t"
        DFMLA(v20,v24,v31,0)
        DFMLA(v21,v25,v31,0)
        DFMLA(v22,v26,v31,0)
        DFMLA(v23,v27,v31,0)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        DFMLA(v0,v24,v29,0)
        DFMLA(v1,v25,v29,0)
        DFMLA(v2,v26,v29,0)
        DFMLA(v3,v27,v29,0)
        "                                   \n\t"
        DFMLA(v4,v24,v29,1)
        DFMLA(v5,v25,v29,1)
        DFMLA(v6,v26,v29,1)
        DFMLA(v7,v27,v29,1)
        "                                   \n\t"
        DFMLA(v8,v24,v30,0)
        DFMLA(v9,v25,v30,0)
        DFMLA(v10,v26,v30,0)
        DFMLA(v11,v27,v30,0)
        "                                   \n\t"
        DFMLA(v12,v24,v30,1)
        DFMLA(v13,v25,v30,1)
        DFMLA(v14,v26,v30,1)
        DFMLA(v15,v27,v30,1)
        "                                   \n\t"
        DFMLA(v16,v24,v31,0)
        DFMLA(v17,v25,v31,0)
        DFMLA(v18,v26,v31,0)
        DFMLA(v19,v27,v31,0)
        "                                   \n\t"
        DFMLA(v20,v24,v31,1)
        DFMLA(v21,v25,v31,1)
        DFMLA(v22,v26,v31,1)
        DFMLA(v23,v27,v31,1)
        "                                   \n\t"
        " sub  %[k], %[k], #3               \n\t"
        " cmp  %[k], #3                     \n\t"
        " b.hs 1b                           \n\t" 
        "                                   \n\t" 
        " 2:                                \n\t" 
        " cbz %[k], 4f                      \n\t" // Take care of the rest if there is some
        "                                   \n\t" 
        " 3:                                \n\t" 
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        LOAD48(v28,v29,v30,%[fac])            
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
        " add %[fac], %[fac], #8            \n\t"
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
        " subs %[k], %[k], 1                \n\t"
        " b.ne 3b                           \n\t"
        "                                   \n\t"
        " 4:                                \n\t"
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
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
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






microkernel_float_armv8a::microkernel_float_armv8a():
microkernel<float> { 6, 16, 64 }
{}


void microkernel_float_armv8a::
euler( const float *x,   const float *y,   const float *z,
             float *r,         float *rinv,
             float *cos_alpha, float *sin_alpha,
             float *cos_beta,  float *sin_beta,
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
        " mov w12, #1                       \n\t" // Integer 1 into w12
        " dup v4.4s, w12                    \n\t" // w12 into v4.s[0],v4.s[1],v4.s[2],v4.s[3]
        " ucvtf v4.4s, v4.4s                \n\t" // Convert v4 into floating point
        " fmov w12, s4                      \n\t" // v4 and w12 now contain the value 1.0
        " mov v5.16b, v4.16b                \n\t" // v4-v7 and w12 now contain floating point 1.0.
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
        FFMUL(v28,v16,v16)
        FFMUL(v29,v17,v17)
        FFMUL(v30,v18,v18)
        FFMUL(v31,v19,v19)
        "                                   \n\t"
        FFMLAV(v28,v20,v20)
        FFMLAV(v29,v21,v21)
        FFMLAV(v30,v22,v22)
        FFMLAV(v31,v23,v23)
        "                                   \n\t"
        FFSQRT(v28,v28)                           // v28-v31 now contain rxy
        FFSQRT(v29,v29)
        FFSQRT(v30,v30)
        FFSQRT(v31,v31)
        "                                   \n\t"
        FFMUL(v0,v24,v24)
        FFMUL(v1,v25,v25)
        FFMUL(v2,v26,v26)
        FFMUL(v3,v27,v27)
        "                                   \n\t"
        FFMLAV(v0,v28,v28)                       
        FFMLAV(v1,v29,v29)
        FFMLAV(v2,v30,v30)
        FFMLAV(v3,v31,v31)
        "                                   \n\t"
        FFSQRT(v0,v0)                             // v0-v3 now contain r = rxyz
        FFSQRT(v1,v1)
        FFSQRT(v2,v2)
        FFSQRT(v3,v3)
        "                                   \n\t"
        STORE64inc(v0,v1,v2,v3,%[r])
        "                                   \n\t"
        " dup v4.4s, w12                    \n\t" 
        " dup v5.4s, w12                    \n\t"
        " dup v6.4s, w12                    \n\t"
        " dup v7.4s, w12                    \n\t"
        "                                   \n\t"
        FFDIV(v4,v4,v0)                           // v4-v7 now contain 1/r
        FFDIV(v5,v5,v1)  
        FFDIV(v6,v6,v2)  
        FFDIV(v7,v7,v3)  
        "                                   \n\t"
        STORE64inc(v4,v5,v6,v7,%[rinv])
        "                                   \n\t"
        FFCMEQZ(v8,v28)                           // rxy == 0 ?
        FFCMEQZ(v9,v29) 
        FFCMEQZ(v10,v30) 
        FFCMEQZ(v11,v31) 
        "                                   \n\t"
        FFDIV(v12,v20,v28)                        // cos_alpha = y/rxy
        FFDIV(v13,v21,v29)                        // no need for y beyond this point.
        FFDIV(v14,v22,v30)   
        FFDIV(v15,v23,v31)   
        "                                   \n\t"
        " bic v12.16b, v12.16b, v8.16b      \n\t" // cos_alpha = 0 if rxy == 0 
        " bic v13.16b, v13.16b, v9.16b      \n\t" 
        " bic v14.16b, v14.16b, v10.16b     \n\t" 
        " bic v15.16b, v15.16b, v11.16b     \n\t" 
        "                                   \n\t"
        " dup v20.4s, w12                   \n\t" // We may overwrite the previous y-register
        " dup v21.4s, w12                   \n\t" // with 1.0.
        " dup v22.4s, w12                   \n\t"
        " dup v23.4s, w12                   \n\t"
        "                                   \n\t"
        " and v20.16b, v20.16b, v8.16b      \n\t" // Contains 1.0 where rxy == 0
        " and v21.16b, v21.16b, v9.16b      \n\t" 
        " and v22.16b, v22.16b, v10.16b     \n\t"
        " and v23.16b, v23.16b, v11.16b     \n\t"
        "                                   \n\t"
        FFADD(v12,v12,v20)                        // cos_alpha = ( rxy == 0 ) ? 1 : y/rxy
        FFADD(v13,v13,v21)
        FFADD(v14,v14,v22)
        FFADD(v15,v15,v23)
        "                                   \n\t"
        STORE64(v12,v13,v14,v15,%[cos_alpha])
        "                                   \n\t"
        FFDIV(v12,v16,v28)                        // sin_alpha = x/rxy
        FFDIV(v13,v17,v29)                        // no need for x beyond this point.
        FFDIV(v14,v18,v30)   
        FFDIV(v15,v19,v31)   
        "                                   \n\t"
        " bic v12.16b, v12.16b, v8.16b      \n\t" // sin_alpha = 0 if rxy == 0 
        " bic v13.16b, v13.16b, v9.16b      \n\t" 
        " bic v14.16b, v14.16b, v10.16b     \n\t" 
        " bic v15.16b, v15.16b, v11.16b     \n\t" 
        "                                   \n\t"
        STORE64(v12,v13,v14,v15,%[sin_alpha])
        "                                   \n\t"
        FFNEG(v28,v28)                            // rxy = -rxy.
        FFNEG(v29,v29)
        FFNEG(v30,v30)
        FFNEG(v31,v31)
        "                                   \n\t"
        FFMUL(v24,v24,v4)                         // cos_beta = z/r
        FFMUL(v25,v25,v5)                   
        FFMUL(v26,v26,v6)                   
        FFMUL(v27,v27,v7)                   
        "                                   \n\t"
        FFMUL(v28,v28,v4)                         // sin_beta = -rxy/r
        FFMUL(v29,v29,v5)                   
        FFMUL(v30,v30,v6)                   
        FFMUL(v31,v31,v7)                   
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
        FFMUL(v8,v8,v0)
        FFMUL(v9,v9,v1)
        FFMUL(v10,v10,v2)
        FFMUL(v11,v11,v3)

        FFMUL(v12,v12,v4)
        FFMUL(v13,v13,v5)
        FFMUL(v14,v14,v6)
        FFMUL(v15,v15,v7)

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

        FFMUL(v8,v16,v0)
        FFMUL(v9,v17,v1)
        FFMUL(v10,v18,v2)
        FFMUL(v11,v19,v3)
        
        FFMUL(v12,v16,v4)
        FFMUL(v13,v17,v5)
        FFMUL(v14,v18,v6)
        FFMUL(v15,v19,v7)

        FFMLSV(v8,v20,v4)
        FFMLSV(v9,v21,v5)
        FFMLSV(v10,v22,v6)
        FFMLSV(v11,v23,v7)
       
        FFMLAV(v12,v20,v0) 
        FFMLAV(v13,v21,v1) 
        FFMLAV(v14,v22,v2) 
        FFMLAV(v15,v23,v3) 

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

        FFMUL(v8,v16,v0)
        FFMUL(v9,v17,v1)
        FFMUL(v10,v18,v2)
        FFMUL(v11,v19,v3)
        
        FFMUL(v12,v16,v4)
        FFMUL(v13,v17,v5)
        FFMUL(v14,v18,v6)
        FFMUL(v15,v19,v7)

        FFMLSV(v8,v20,v4)
        FFMLSV(v9,v21,v5)
        FFMLSV(v10,v22,v6)
        FFMLSV(v11,v23,v7)
       
        FFMLAV(v12,v20,v0) 
        FFMLAV(v13,v21,v1) 
        FFMLAV(v14,v22,v2) 
        FFMLAV(v15,v23,v3) 

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

void microkernel_float_armv8a::
rotscale( const float *cos,
          const float *sin,
          const float *scale,
          const float *real_in,
          const float *imag_in,
                float *real_out,
                float *imag_out,
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

          FFMUL(v20,v0,v8)                        // real_out = cos*real_in
          FFMUL(v21,v1,v9)
          FFMUL(v22,v2,v10)
          FFMUL(v23,v3,v11)

          FFMUL(v24,v0,v12)                       // imag_out = cos*imag_in
          FFMUL(v25,v1,v13)
          FFMUL(v26,v2,v14)
          FFMUL(v27,v3,v15)

          FFMLSV(v20,v4,v12)                      // real_out -= sin*imag_in
          FFMLSV(v21,v5,v13)
          FFMLSV(v22,v6,v14)
          FFMLSV(v23,v7,v15)

          FFMLAV(v24,v4,v8)                       // imag_out += sin*real_in
          FFMLAV(v25,v5,v9)
          FFMLAV(v26,v6,v10)
          FFMLAV(v27,v7,v11)

          FFMUL(v20,v20,v16)                      // real_out *= fac
          FFMUL(v21,v21,v17)                       
          FFMUL(v22,v22,v18)                       
          FFMUL(v23,v23,v19)                       

          FFMUL(v24,v24,v16)                      // imag_out *= fac
          FFMUL(v25,v25,v17)                       
          FFMUL(v26,v26,v18)                       
          FFMUL(v27,v27,v19)                       

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
        
          FFMUL(v20,v0,v8)                        // real_out = cos*real_in
          FFMUL(v21,v1,v9)
          FFMUL(v22,v2,v10)
          FFMUL(v23,v3,v11)

          FFMUL(v24,v0,v12)                       // imag_out = cos*imag_in
          FFMUL(v25,v1,v13)
          FFMUL(v26,v2,v14)
          FFMUL(v27,v3,v15)

          FFMLAV(v20,v4,v12)                      // real_out += sin*imag_in
          FFMLAV(v21,v5,v13)
          FFMLAV(v22,v6,v14)
          FFMLAV(v23,v7,v15)

          FFMLSV(v24,v4,v8)                       // imag_out -= sin*real_in
          FFMLSV(v25,v5,v9)
          FFMLSV(v26,v6,v10)
          FFMLSV(v27,v7,v11)

          FFMUL(v20,v20,v16)                      // real_out *= fac
          FFMUL(v21,v21,v17)                       
          FFMUL(v22,v22,v18)                       
          FFMUL(v23,v23,v19)                       

          FFMUL(v24,v24,v16)                      // imag_out *= fac
          FFMUL(v25,v25,v17)                       
          FFMUL(v26,v26,v18)                       
          FFMUL(v27,v27,v19)                       

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

void microkernel_float_armv8a::
swap( const float *mat,
      const float *in,
            float *out,
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
        " ldr q28, [%[mat]]                 \n\t" // Load the next two columns of swap matrix
        " ldr d29, [%[mat],#16]             \n\t"
        " add %[mat], %[mat], #24           \n\t"
        "                                   \n\t"
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v8,v24,v28,1)
        FFMLA(v9,v25,v28,1)
        FFMLA(v10,v26,v28,1)
        FFMLA(v11,v27,v28,1)
        "                                   \n\t"
        FFMLA(v16,v24,v28,2)
        FFMLA(v17,v25,v28,2)
        FFMLA(v18,v26,v28,2)
        FFMLA(v19,v27,v28,2)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])        // Load next row of input
        "                                   \n\t"
        FFMLA(v4,v24,v28,3)
        FFMLA(v5,v25,v28,3)
        FFMLA(v6,v26,v28,3)
        FFMLA(v7,v27,v28,3)
        "                                   \n\t"
        FFMLA(v12,v24,v29,0)
        FFMLA(v13,v25,v29,0)
        FFMLA(v14,v26,v29,0)
        FFMLA(v15,v27,v29,0)
        "                                   \n\t"
        FFMLA(v20,v24,v29,1)
        FFMLA(v21,v25,v29,1)
        FFMLA(v22,v26,v29,1)
        FFMLA(v23,v27,v29,1)
        "                                   \n\t"
        " subs %[k], %[k], #1               \n\t"
        " b.ne 1b                           \n\t" // while ( k-- );
        "                                   \n\t"
        " 2:                                \n\t"
        " cbz %[k_odd], 3f                  \n\t" // if ( k_odd )
        LOAD64(v24,v25,v26,v27,%[in])             // Load next row of input
        " ldr d28, [%[mat]]                 \n\t" // Load the last column of swap matrix.
        " ldr s29, [%[mat],#8]              \n\t"
        "                                   \n\t"
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v8,v24,v28,1)
        FFMLA(v9,v25,v28,1)
        FFMLA(v10,v26,v28,1)
        FFMLA(v11,v27,v28,1)
        "                                   \n\t"
        FFMLA(v16,v24,v29,0)
        FFMLA(v17,v25,v29,0)
        FFMLA(v18,v26,v29,0)
        FFMLA(v19,v27,v29,0)
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
        "v24", "v25", "v26", "v27", "v28", "v29"
    );
}

void microkernel_float_armv8a::
zm2l( const float *fac,
      const float *in,
            float *out,
      size_t k, bool pattern ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
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
        " cmp  %[k], #3                     \n\t"
        " b.lo 2f                           \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ... One iteration does three columns
        "                                   \n\t" // of the faculty matrix
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        " ldp q28, q29, [%[fac]]            \n\t" // Load faculties for next three columns
        "                                   \n\t" 
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v4,v24,v28,1)
        FFMLA(v5,v25,v28,1)
        FFMLA(v6,v26,v28,1)
        FFMLA(v7,v27,v28,1)
        "                                   \n\t"
        FFMLA(v8,v24,v28,2)
        FFMLA(v9,v25,v28,2)
        FFMLA(v10,v26,v28,2)
        FFMLA(v11,v27,v28,2)
        "                                   \n\t"
        " add %[fac], %[fac], #12           \n\t"
        "                                   \n\t"
        FFMLA(v12,v24,v28,3)
        FFMLA(v13,v25,v28,3)
        FFMLA(v14,v26,v28,3)
        FFMLA(v15,v27,v28,3)
        "                                   \n\t"
        FFMLA(v16,v24,v29,0)
        FFMLA(v17,v25,v29,0)
        FFMLA(v18,v26,v29,0)
        FFMLA(v19,v27,v29,0)
        "                                   \n\t"
        FFMLA(v20,v24,v29,1)
        FFMLA(v21,v25,v29,1)
        FFMLA(v22,v26,v29,1)
        FFMLA(v23,v27,v29,1)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        FFMLA(v0,v24,v28,1)
        FFMLA(v1,v25,v28,1)
        FFMLA(v2,v26,v28,1)
        FFMLA(v3,v27,v28,1)
        "                                   \n\t"
        FFMLA(v4,v24,v28,2)
        FFMLA(v5,v25,v28,2)
        FFMLA(v6,v26,v28,2)
        FFMLA(v7,v27,v28,2)
        "                                   \n\t"
        FFMLA(v8,v24,v28,3)
        FFMLA(v9,v25,v28,3)
        FFMLA(v10,v26,v28,3)
        FFMLA(v11,v27,v28,3)
        "                                   \n\t"
        FFMLA(v12,v24,v29,0)
        FFMLA(v13,v25,v29,0)
        FFMLA(v14,v26,v29,0)
        FFMLA(v15,v27,v29,0)
        "                                   \n\t"
        FFMLA(v16,v24,v29,1)
        FFMLA(v17,v25,v29,1)
        FFMLA(v18,v26,v29,1)
        FFMLA(v19,v27,v29,1)
        "                                   \n\t"
        FFMLA(v20,v24,v29,2)
        FFMLA(v21,v25,v29,2)
        FFMLA(v22,v26,v29,2)
        FFMLA(v23,v27,v29,2)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        FFMLA(v0,v24,v28,2)
        FFMLA(v1,v25,v28,2)
        FFMLA(v2,v26,v28,2)
        FFMLA(v3,v27,v28,2)
        "                                   \n\t"
        FFMLA(v4,v24,v28,3)
        FFMLA(v5,v25,v28,3)
        FFMLA(v6,v26,v28,3)
        FFMLA(v7,v27,v28,3)
        "                                   \n\t"
        FFMLA(v8,v24,v29,0)
        FFMLA(v9,v25,v29,0)
        FFMLA(v10,v26,v29,0)
        FFMLA(v11,v27,v29,0)
        "                                   \n\t"
        FFMLA(v12,v24,v29,1)
        FFMLA(v13,v25,v29,1)
        FFMLA(v14,v26,v29,1)
        FFMLA(v15,v27,v29,1)
        "                                   \n\t"
        FFMLA(v16,v24,v29,2)
        FFMLA(v17,v25,v29,2)
        FFMLA(v18,v26,v29,2)
        FFMLA(v19,v27,v29,2)
        "                                   \n\t"
        FFMLA(v20,v24,v29,3)
        FFMLA(v21,v25,v29,3)
        FFMLA(v22,v26,v29,3)
        FFMLA(v23,v27,v29,3)
        "                                   \n\t"
        " sub  %[k], %[k], #3               \n\t"
        " cmp  %[k], #3                     \n\t"
        " b.hs 1b                           \n\t" // while ( --k );
        "                                   \n\t" 
        " 2:                                \n\t" 
        " cbz %[k], 4f                      \n\t" // Take care of the rest if there is some
        "                                   \n\t" 
        " 3:                                \n\t" 
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        " ldr q28, [%[fac]]                 \n\t"
        " ldr d29, [%[fac],#16]             \n\t"
        "                                   \n\t"
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v4,v24,v28,1)
        FFMLA(v5,v25,v28,1)
        FFMLA(v6,v26,v28,1)
        FFMLA(v7,v27,v28,1)
        "                                   \n\t"
        FFMLA(v8,v24,v28,2)
        FFMLA(v9,v25,v28,2)
        FFMLA(v10,v26,v28,2)
        FFMLA(v11,v27,v28,2)
        "                                   \n\t"
        " add %[fac], %[fac], #4            \n\t"
        "                                   \n\t"
        FFMLA(v12,v24,v28,3)
        FFMLA(v13,v25,v28,3)
        FFMLA(v14,v26,v28,3)
        FFMLA(v15,v27,v28,3)
        "                                   \n\t"
        FFMLA(v16,v24,v29,0)
        FFMLA(v17,v25,v29,0)
        FFMLA(v18,v26,v29,0)
        FFMLA(v19,v27,v29,0)
        "                                   \n\t"
        FFMLA(v20,v24,v29,1)
        FFMLA(v21,v25,v29,1)
        FFMLA(v22,v26,v29,1)
        FFMLA(v23,v27,v29,1)
        "                                   \n\t"
        " subs %[k], %[k], 1                \n\t"
        " b.ne 3b                           \n\t"
        "                                   \n\t"
        " 4:                                \n\t"
        " cbz %w[pattern], 5f               \n\t" // if ( pattern )
        FFNEG(v0,v0)
        FFNEG(v1,v1)
        FFNEG(v2,v2)
        FFNEG(v3,v3)
        FFNEG(v8,v8)
        FFNEG(v9,v9)
        FFNEG(v10,v10)
        FFNEG(v11,v11)
        FFNEG(v16,v16)
        FFNEG(v17,v17)
        FFNEG(v18,v18)
        FFNEG(v19,v19)
        " b 6f                              \n\t" 
        " 5:                                \n\t" // else:
        FFNEG(v4,v4)
        FFNEG(v5,v5)
        FFNEG(v6,v6)
        FFNEG(v7,v7)
        FFNEG(v12,v12)
        FFNEG(v13,v13)
        FFNEG(v14,v14)
        FFNEG(v15,v15)
        FFNEG(v20,v20)
        FFNEG(v21,v21)
        FFNEG(v22,v22)
        FFNEG(v23,v23)
        "                                   \n\t" 
        " 6:                                \n\t" // Store result.
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
        "v24", "v25", "v26", "v27", "v28", "v29"
    ); 
}

void microkernel_float_armv8a::
zm2m( const float *fac,
      const float *in,
            float *out,
            size_t k ) const noexcept
{
    if ( k == 0 ) return;

    __asm__ volatile
    (
        "                                   \n\t"
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
        " cmp  %[k], #3                     \n\t"
        " b.lo 2f                           \n\t"
        "                                   \n\t"
        ".align 5                           \n\t" 
        " 1:                                \n\t" // do ... One iteration does three columns
        "                                   \n\t" // of the faculty matrix
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        " ldp q28, q29, [%[fac]]            \n\t" // Load faculties for next three columns
        "                                   \n\t" 
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v4,v24,v28,1)
        FFMLA(v5,v25,v28,1)
        FFMLA(v6,v26,v28,1)
        FFMLA(v7,v27,v28,1)
        "                                   \n\t"
        FFMLA(v8,v24,v28,2)
        FFMLA(v9,v25,v28,2)
        FFMLA(v10,v26,v28,2)
        FFMLA(v11,v27,v28,2)
        "                                   \n\t"
        " add %[fac], %[fac], #12           \n\t"
        "                                   \n\t"
        FFMLA(v12,v24,v28,3)
        FFMLA(v13,v25,v28,3)
        FFMLA(v14,v26,v28,3)
        FFMLA(v15,v27,v28,3)
        "                                   \n\t"
        FFMLA(v16,v24,v29,0)
        FFMLA(v17,v25,v29,0)
        FFMLA(v18,v26,v29,0)
        FFMLA(v19,v27,v29,0)
        "                                   \n\t"
        FFMLA(v20,v24,v29,1)
        FFMLA(v21,v25,v29,1)
        FFMLA(v22,v26,v29,1)
        FFMLA(v23,v27,v29,1)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        FFMLA(v0,v24,v28,1)
        FFMLA(v1,v25,v28,1)
        FFMLA(v2,v26,v28,1)
        FFMLA(v3,v27,v28,1)
        "                                   \n\t"
        FFMLA(v4,v24,v28,2)
        FFMLA(v5,v25,v28,2)
        FFMLA(v6,v26,v28,2)
        FFMLA(v7,v27,v28,2)
        "                                   \n\t"
        FFMLA(v8,v24,v28,3)
        FFMLA(v9,v25,v28,3)
        FFMLA(v10,v26,v28,3)
        FFMLA(v11,v27,v28,3)
        "                                   \n\t"
        FFMLA(v12,v24,v29,0)
        FFMLA(v13,v25,v29,0)
        FFMLA(v14,v26,v29,0)
        FFMLA(v15,v27,v29,0)
        "                                   \n\t"
        FFMLA(v16,v24,v29,1)
        FFMLA(v17,v25,v29,1)
        FFMLA(v18,v26,v29,1)
        FFMLA(v19,v27,v29,1)
        "                                   \n\t"
        FFMLA(v20,v24,v29,2)
        FFMLA(v21,v25,v29,2)
        FFMLA(v22,v26,v29,2)
        FFMLA(v23,v27,v29,2)
        "                                   \n\t"
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        "                                   \n\t"
        FFMLA(v0,v24,v28,2)
        FFMLA(v1,v25,v28,2)
        FFMLA(v2,v26,v28,2)
        FFMLA(v3,v27,v28,2)
        "                                   \n\t"
        FFMLA(v4,v24,v28,3)
        FFMLA(v5,v25,v28,3)
        FFMLA(v6,v26,v28,3)
        FFMLA(v7,v27,v28,3)
        "                                   \n\t"
        FFMLA(v8,v24,v29,0)
        FFMLA(v9,v25,v29,0)
        FFMLA(v10,v26,v29,0)
        FFMLA(v11,v27,v29,0)
        "                                   \n\t"
        FFMLA(v12,v24,v29,1)
        FFMLA(v13,v25,v29,1)
        FFMLA(v14,v26,v29,1)
        FFMLA(v15,v27,v29,1)
        "                                   \n\t"
        FFMLA(v16,v24,v29,2)
        FFMLA(v17,v25,v29,2)
        FFMLA(v18,v26,v29,2)
        FFMLA(v19,v27,v29,2)
        "                                   \n\t"
        FFMLA(v20,v24,v29,3)
        FFMLA(v21,v25,v29,3)
        FFMLA(v22,v26,v29,3)
        FFMLA(v23,v27,v29,3)
        "                                   \n\t"
        " sub  %[k], %[k], #3               \n\t"
        " cmp  %[k], #3                     \n\t"
        " b.hs 1b                           \n\t" 
        "                                   \n\t" 
        " 2:                                \n\t" 
        " cbz %[k], 4f                      \n\t" // Take care of the rest if there is some
        "                                   \n\t" 
        " 3:                                \n\t" 
        LOAD64inc(v24,v25,v26,v27,%[in])          // Load next line of input
        " ldr q28, [%[fac]]                 \n\t"
        " ldr d29, [%[fac],#16]             \n\t"
        "                                   \n\t" 
        FFMLA(v0,v24,v28,0)
        FFMLA(v1,v25,v28,0)
        FFMLA(v2,v26,v28,0)
        FFMLA(v3,v27,v28,0)
        "                                   \n\t"
        FFMLA(v4,v24,v28,1)
        FFMLA(v5,v25,v28,1)
        FFMLA(v6,v26,v28,1)
        FFMLA(v7,v27,v28,1)
        "                                   \n\t"
        FFMLA(v8,v24,v28,2)
        FFMLA(v9,v25,v28,2)
        FFMLA(v10,v26,v28,2)
        FFMLA(v11,v27,v28,2)
        "                                   \n\t"
        " add %[fac], %[fac], #4            \n\t"
        "                                   \n\t"
        FFMLA(v12,v24,v28,3)
        FFMLA(v13,v25,v28,3)
        FFMLA(v14,v26,v28,3)
        FFMLA(v15,v27,v28,3)
        "                                   \n\t"
        FFMLA(v16,v24,v29,0)
        FFMLA(v17,v25,v29,0)
        FFMLA(v18,v26,v29,0)
        FFMLA(v19,v27,v29,0)
        "                                   \n\t"
        FFMLA(v20,v24,v29,1)
        FFMLA(v21,v25,v29,1)
        FFMLA(v22,v26,v29,1)
        FFMLA(v23,v27,v29,1)
        "                                   \n\t"
        " subs %[k], %[k], 1                \n\t"
        " b.ne 3b                           \n\t"
        "                                   \n\t"
        " 4:                                \n\t"
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
        "v24", "v25", "v26", "v27", "v28", "v29" 
    ); 
}

void microkernel_float_armv8a::
swap2trans_buf( const float  *real_in, 
                const float  *imag_in,
                      float **real_out, 
                      float **imag_out,
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

void microkernel_float_armv8a::
trans2swap_buf( const float *const *const  real_in, 
                const float *const *const  imag_in,
                      float               *real_out, 
                      float               *imag_out,
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

void microkernel_float_armv8a::
solid2buf( const float *const *p_solids,
           const float *const  zeros,
           const size_t       *P,
                 float        *real_out,
                 float        *imag_out,
                 size_t        n ) const noexcept
{
    const float* solids[16] =
    {
        ( n < P[ 0] ) ? ( p_solids[ 0] + (n  )*(n+1) ) : zeros,
        ( n < P[ 1] ) ? ( p_solids[ 1] + (n  )*(n+1) ) : zeros,
        ( n < P[ 2] ) ? ( p_solids[ 2] + (n  )*(n+1) ) : zeros,
        ( n < P[ 3] ) ? ( p_solids[ 3] + (n  )*(n+1) ) : zeros,
        ( n < P[ 4] ) ? ( p_solids[ 4] + (n  )*(n+1) ) : zeros,
        ( n < P[ 5] ) ? ( p_solids[ 5] + (n  )*(n+1) ) : zeros,
        ( n < P[ 6] ) ? ( p_solids[ 6] + (n  )*(n+1) ) : zeros,
        ( n < P[ 7] ) ? ( p_solids[ 7] + (n  )*(n+1) ) : zeros,
        ( n < P[ 8] ) ? ( p_solids[ 8] + (n  )*(n+1) ) : zeros,
        ( n < P[ 9] ) ? ( p_solids[ 9] + (n  )*(n+1) ) : zeros,
        ( n < P[10] ) ? ( p_solids[10] + (n  )*(n+1) ) : zeros,
        ( n < P[11] ) ? ( p_solids[11] + (n  )*(n+1) ) : zeros,
        ( n < P[12] ) ? ( p_solids[12] + (n  )*(n+1) ) : zeros,
        ( n < P[13] ) ? ( p_solids[13] + (n  )*(n+1) ) : zeros,
        ( n < P[14] ) ? ( p_solids[14] + (n  )*(n+1) ) : zeros,
        ( n < P[15] ) ? ( p_solids[15] + (n  )*(n+1) ) : zeros
    };

    size_t m = n+1; // Number of elements we actually need to copy.

    __asm__ volatile
    (
        "                                     \n\t" // First we take care of the real parts.
        " mov x16, %[m]                       \n\t" // Initialise loop variable.
        "                                     \n\t"
        " ldr x0, %[solids], #8               \n\t" // Load addresses of real parts.
        " ldr x1, %[solids], #8               \n\t"
        " ldr x2, %[solids], #8               \n\t"
        " ldr x3, %[solids], #8               \n\t"
        " ldr x4, %[solids], #8               \n\t"
        " ldr x5, %[solids], #8               \n\t"
        " ldr x6, %[solids], #8               \n\t"
        " ldr x7, %[solids], #8               \n\t"
        " ldr x8, %[solids], #8               \n\t"
        " ldr x9, %[solids], #8               \n\t"
        " ldr x10, %[solids], #8              \n\t"
        " ldr x11, %[solids], #8              \n\t"
        " ldr x12, %[solids], #8              \n\t"
        " ldr x13, %[solids], #8              \n\t"
        " ldr x14, %[solids], #8              \n\t"
        " ldr x15, %[solids], #8              \n\t"
        "                                     \n\t"
        " cmp x16, #4                         \n\t" // The following loop copies 4 elements per iteration
        " b.lo 1f                             \n\t" // Skip if we have less than four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 2:                                  \n\t"
        " ldr q16, [x0], #16                  \n\t"
        " ldr q17, [x1], #16                  \n\t"
        " ldr q18, [x2], #16                  \n\t"
        " ldr q19, [x3], #16                  \n\t"
        " ldr q20, [x4], #16                  \n\t"
        " ldr q21, [x5], #16                  \n\t"
        " ldr q22, [x6], #16                  \n\t"
        " ldr q23, [x7], #16                  \n\t"
        " ldr q24, [x8], #16                  \n\t"
        " ldr q25, [x9], #16                  \n\t"
        " ldr q26, [x10], #16                 \n\t"
        " ldr q27, [x11], #16                 \n\t"
        " ldr q28, [x12], #16                 \n\t"
        " ldr q29, [x13], #16                 \n\t"
        " ldr q30, [x14], #16                 \n\t"
        " ldr q31, [x15], #16                 \n\t"
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,0,%[real_out])
        FST4inc(v20,v21,v22,v23,0,%[real_out])
        FST4inc(v24,v25,v26,v27,0,%[real_out])
        FST4inc(v28,v29,v30,v31,0,%[real_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,1,%[real_out])
        FST4inc(v20,v21,v22,v23,1,%[real_out])
        FST4inc(v24,v25,v26,v27,1,%[real_out])
        FST4inc(v28,v29,v30,v31,1,%[real_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,2,%[real_out])
        FST4inc(v20,v21,v22,v23,2,%[real_out])
        FST4inc(v24,v25,v26,v27,2,%[real_out])
        FST4inc(v28,v29,v30,v31,2,%[real_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,3,%[real_out])
        FST4inc(v20,v21,v22,v23,3,%[real_out])
        FST4inc(v24,v25,v26,v27,3,%[real_out])
        FST4inc(v28,v29,v30,v31,3,%[real_out])
        "                                     \n\t"
        " sub x16, x16, #4                    \n\t"
        " cmp x16, #4                         \n\t"
        " b.hs 2b                             \n\t"
        "                                     \n\t"
        " 1:                                  \n\t"
        " cbz x16, 0f                         \n\t"
        " ldr s16, [x0], #4                   \n\t"
        " ldr s17, [x1], #4                   \n\t"
        " ldr s18, [x2], #4                   \n\t"
        " ldr s19, [x3], #4                   \n\t"
        " ldr s20, [x4], #4                   \n\t"
        " ldr s21, [x5], #4                   \n\t"
        " ldr s22, [x6], #4                   \n\t"
        " ldr s23, [x7], #4                   \n\t"
        " ldr s24, [x8], #4                   \n\t"
        " ldr s25, [x9], #4                   \n\t"
        " ldr s26, [x10], #4                  \n\t"
        " ldr s27, [x11], #4                  \n\t"
        " ldr s28, [x12], #4                  \n\t"
        " ldr s29, [x13], #4                  \n\t"
        " ldr s30, [x14], #4                  \n\t"
        " ldr s31, [x15], #4                  \n\t"
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,0,%[real_out])
        FST4inc(v20,v21,v22,v23,0,%[real_out])
        FST4inc(v24,v25,v26,v27,0,%[real_out])
        FST4inc(v28,v29,v30,v31,0,%[real_out])
        "                                     \n\t"
        " subs x16, x16, 1                    \n\t"
        " b.ne 1b                             \n\t"
        "                                     \n\t"
        "                                     \n\t"
        " 0:                                  \n\t" // Take care of imaginary parts.
        "                                     \n\t" // Same code as above, just different output location.
        " mov x16, %[m]                       \n\t" // Initialise loop variable.
        "                                     \n\t"
        " cmp x16, #4                         \n\t" // The following loop copies 4 elements per iteration
        " b.lo 1f                             \n\t" // Skip if we have less than four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 2:                                  \n\t"
        " ldr q16, [x0], #16                  \n\t"
        " ldr q17, [x1], #16                  \n\t"
        " ldr q18, [x2], #16                  \n\t"
        " ldr q19, [x3], #16                  \n\t"
        " ldr q20, [x4], #16                  \n\t"
        " ldr q21, [x5], #16                  \n\t"
        " ldr q22, [x6], #16                  \n\t"
        " ldr q23, [x7], #16                  \n\t"
        " ldr q24, [x8], #16                  \n\t"
        " ldr q25, [x9], #16                  \n\t"
        " ldr q26, [x10], #16                 \n\t"
        " ldr q27, [x11], #16                 \n\t"
        " ldr q28, [x12], #16                 \n\t"
        " ldr q29, [x13], #16                 \n\t"
        " ldr q30, [x14], #16                 \n\t"
        " ldr q31, [x15], #16                 \n\t"
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,0,%[imag_out])
        FST4inc(v20,v21,v22,v23,0,%[imag_out])
        FST4inc(v24,v25,v26,v27,0,%[imag_out])
        FST4inc(v28,v29,v30,v31,0,%[imag_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,1,%[imag_out])
        FST4inc(v20,v21,v22,v23,1,%[imag_out])
        FST4inc(v24,v25,v26,v27,1,%[imag_out])
        FST4inc(v28,v29,v30,v31,1,%[imag_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,2,%[imag_out])
        FST4inc(v20,v21,v22,v23,2,%[imag_out])
        FST4inc(v24,v25,v26,v27,2,%[imag_out])
        FST4inc(v28,v29,v30,v31,2,%[imag_out])
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,3,%[imag_out])
        FST4inc(v20,v21,v22,v23,3,%[imag_out])
        FST4inc(v24,v25,v26,v27,3,%[imag_out])
        FST4inc(v28,v29,v30,v31,3,%[imag_out])
        "                                     \n\t"
        " sub x16, x16, #4                    \n\t"
        " cmp x16, #4                         \n\t"
        " b.hs 2b                             \n\t"
        "                                     \n\t"
        " 1:                                  \n\t"
        " cbz x16, 0f                         \n\t"
        " ldr s16, [x0], #4                   \n\t"
        " ldr s17, [x1], #4                   \n\t"
        " ldr s18, [x2], #4                   \n\t"
        " ldr s19, [x3], #4                   \n\t"
        " ldr s20, [x4], #4                   \n\t"
        " ldr s21, [x5], #4                   \n\t"
        " ldr s22, [x6], #4                   \n\t"
        " ldr s23, [x7], #4                   \n\t"
        " ldr s24, [x8], #4                   \n\t"
        " ldr s25, [x9], #4                   \n\t"
        " ldr s26, [x10], #4                  \n\t"
        " ldr s27, [x11], #4                  \n\t"
        " ldr s28, [x12], #4                  \n\t"
        " ldr s29, [x13], #4                  \n\t"
        " ldr s30, [x14], #4                  \n\t"
        " ldr s31, [x15], #4                  \n\t"
        "                                     \n\t"
        FST4inc(v16,v17,v18,v19,0,%[imag_out])
        FST4inc(v20,v21,v22,v23,0,%[imag_out])
        FST4inc(v24,v25,v26,v27,0,%[imag_out])
        FST4inc(v28,v29,v30,v31,0,%[imag_out])
        "                                     \n\t"
        " subs x16, x16, 1                    \n\t"
        " b.ne 1b                             \n\t"
        "                                     \n\t"
        "                                     \n\t"
        " 0:                                  \n\t" // Take care of imaginary parts.
        "                                     \n\t" // Same code as above, just different output location.
        : // Output
            [real_out] "+r"(real_out),
            [imag_out] "+r"(imag_out),
            [solids]   "+m" (solids)

        : // Input
            [m]        "r"(m)

        : // Clobbered registers
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
        "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
        "x16",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
}

void microkernel_float_armv8a::
buf2solid( const float  *real_in, 
           const float  *imag_in, 
                 float **p_solids,
                 float  *trash, 
           const size_t  *P,
                 size_t   n ) const noexcept
{
    float* solids[ 16 ] =
    {
        ( n < P[ 0] ) ? ( p_solids[ 0] + (n  )*(n+1) ) : trash,
        ( n < P[ 1] ) ? ( p_solids[ 1] + (n  )*(n+1) ) : trash,
        ( n < P[ 2] ) ? ( p_solids[ 2] + (n  )*(n+1) ) : trash,
        ( n < P[ 3] ) ? ( p_solids[ 3] + (n  )*(n+1) ) : trash,
        ( n < P[ 4] ) ? ( p_solids[ 4] + (n  )*(n+1) ) : trash,
        ( n < P[ 5] ) ? ( p_solids[ 5] + (n  )*(n+1) ) : trash,
        ( n < P[ 6] ) ? ( p_solids[ 6] + (n  )*(n+1) ) : trash,
        ( n < P[ 7] ) ? ( p_solids[ 7] + (n  )*(n+1) ) : trash,
        ( n < P[ 8] ) ? ( p_solids[ 8] + (n  )*(n+1) ) : trash,
        ( n < P[ 9] ) ? ( p_solids[ 9] + (n  )*(n+1) ) : trash,
        ( n < P[10] ) ? ( p_solids[10] + (n  )*(n+1) ) : trash,
        ( n < P[11] ) ? ( p_solids[11] + (n  )*(n+1) ) : trash,
        ( n < P[12] ) ? ( p_solids[12] + (n  )*(n+1) ) : trash,
        ( n < P[13] ) ? ( p_solids[13] + (n  )*(n+1) ) : trash,
        ( n < P[14] ) ? ( p_solids[14] + (n  )*(n+1) ) : trash,
        ( n < P[15] ) ? ( p_solids[15] + (n  )*(n+1) ) : trash
    };

    size_t m = n+1; // Number of elements we actually need to copy.

    __asm__ volatile
    (
        "                                     \n\t" // First we take care of the real parts.
        " mov x16, %[m]                       \n\t" // Initialise loop variable.
        "                                     \n\t"
        " ldr x0, %[solids], #8               \n\t" // Load addresses of real parts.
        " ldr x1, %[solids], #8               \n\t"
        " ldr x2, %[solids], #8               \n\t"
        " ldr x3, %[solids], #8               \n\t"
        " ldr x4, %[solids], #8               \n\t"
        " ldr x5, %[solids], #8               \n\t"
        " ldr x6, %[solids], #8               \n\t"
        " ldr x7, %[solids], #8               \n\t"
        " ldr x8, %[solids], #8               \n\t"
        " ldr x9, %[solids], #8               \n\t"
        " ldr x10, %[solids], #8              \n\t"
        " ldr x11, %[solids], #8              \n\t"
        " ldr x12, %[solids], #8              \n\t"
        " ldr x13, %[solids], #8              \n\t"
        " ldr x14, %[solids], #8              \n\t"
        " ldr x15, %[solids], #8              \n\t"
        "                                     \n\t"
        " cmp x16, #4                         \n\t" // The following loop copies 4 elements per iteration
        " b.lo 1f                             \n\t" // Skip if we have less than four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 2:                                  \n\t"
        FLD4inc(v16,v17,v18,v19,0,%[real_in])
        FLD4inc(v20,v21,v22,v23,0,%[real_in])
        FLD4inc(v24,v25,v26,v27,0,%[real_in])
        FLD4inc(v28,v29,v30,v31,0,%[real_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,1,%[real_in])
        FLD4inc(v20,v21,v22,v23,1,%[real_in])
        FLD4inc(v24,v25,v26,v27,1,%[real_in])
        FLD4inc(v28,v29,v30,v31,1,%[real_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,2,%[real_in])
        FLD4inc(v20,v21,v22,v23,2,%[real_in])
        FLD4inc(v24,v25,v26,v27,2,%[real_in])
        FLD4inc(v28,v29,v30,v31,2,%[real_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,3,%[real_in])
        FLD4inc(v20,v21,v22,v23,3,%[real_in])
        FLD4inc(v24,v25,v26,v27,3,%[real_in])
        FLD4inc(v28,v29,v30,v31,3,%[real_in])
        "                                     \n\t"
        " ldr q0, [x0]                        \n\t"
        FFADD(v0,v16,v0)
        " str q0, [x0], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x1]                        \n\t"
        FFADD(v0,v17,v0)
        " str q0, [x1], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x2]                        \n\t"
        FFADD(v0,v18,v0)
        " str q0, [x2], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x3]                        \n\t"
        FFADD(v0,v19,v0)
        " str q0, [x3], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x4]                        \n\t"
        FFADD(v0,v20,v0)
        " str q0, [x4], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x5]                        \n\t"
        FFADD(v0,v21,v0)
        " str q0, [x5], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x6]                        \n\t"
        FFADD(v0,v22,v0)
        " str q0, [x6], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x7]                        \n\t"
        FFADD(v0,v23,v0)
        " str q0, [x7], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x8]                        \n\t"
        FFADD(v0,v24,v0)
        " str q0, [x8], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x9]                        \n\t"
        FFADD(v0,v25,v0)
        " str q0, [x9], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x10]                       \n\t"
        FFADD(v0,v26,v0)
        " str q0, [x10], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x11]                       \n\t"
        FFADD(v0,v27,v0)
        " str q0, [x11], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x12]                       \n\t"
        FFADD(v0,v28,v0)
        " str q0, [x12], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x13]                       \n\t"
        FFADD(v0,v29,v0)
        " str q0, [x13], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x14]                       \n\t"
        FFADD(v0,v30,v0)
        " str q0, [x14], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x15]                       \n\t"
        FFADD(v0,v31,v0)
        " str q0, [x15], #16                  \n\t"
        "                                     \n\t"
        " sub x16, x16, #4                    \n\t"
        " cmp x16, #4                         \n\t"
        " b.hs 2b                             \n\t"
        "                                     \n\t"
        " cbz x16, 0f                         \n\t"
        " 1:                                  \n\t"    
        " ldr s16, [%[real_in]], #4           \n\t"
        " ldr s17, [%[real_in]], #4           \n\t"
        " ldr s18, [%[real_in]], #4           \n\t"
        " ldr s19, [%[real_in]], #4           \n\t"
        " ldr s20, [%[real_in]], #4           \n\t"
        " ldr s21, [%[real_in]], #4           \n\t"
        " ldr s22, [%[real_in]], #4           \n\t"
        " ldr s23, [%[real_in]], #4           \n\t"
        " ldr s24, [%[real_in]], #4           \n\t"
        " ldr s25, [%[real_in]], #4           \n\t"
        " ldr s26, [%[real_in]], #4           \n\t"
        " ldr s27, [%[real_in]], #4           \n\t"
        " ldr s28, [%[real_in]], #4           \n\t"
        " ldr s29, [%[real_in]], #4           \n\t"
        " ldr s30, [%[real_in]], #4           \n\t"
        " ldr s31, [%[real_in]], #4           \n\t"
        "                                     \n\t"
        " ldr s0, [x0]                        \n\t"
        " fadd s0,s16,s0                      \n\t"
        " str s0, [x0], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x1]                        \n\t"
        " fadd s0,s17,s0                      \n\t"
        " str s0, [x1], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x2]                        \n\t"
        " fadd s0,s18,s0                      \n\t"
        " str s0, [x2], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x3]                        \n\t"
        " fadd s0,s19,s0                      \n\t"
        " str s0, [x3], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x4]                        \n\t"
        " fadd s0,s20,s0                      \n\t"
        " str s0, [x4], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x5]                        \n\t"
        " fadd s0,s21,s0                      \n\t"
        " str s0, [x5], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x6]                        \n\t"
        " fadd s0,s22,s0                      \n\t"
        " str s0, [x6], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x7]                        \n\t"
        " fadd s0,s23,s0                      \n\t"
        " str s0, [x7], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x8]                        \n\t"
        " fadd s0,s24,s0                      \n\t"
        " str s0, [x8], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x9]                        \n\t"
        " fadd s0,s25,s0                      \n\t"
        " str s0, [x9], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x10]                       \n\t"
        " fadd s0,s26,s0                      \n\t"
        " str s0, [x10], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x11]                       \n\t"
        " fadd s0,s27,s0                      \n\t"
        " str s0, [x11], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x12]                       \n\t"
        " fadd s0,s28,s0                      \n\t"
        " str s0, [x12], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x13]                       \n\t"
        " fadd s0,s29,s0                      \n\t"
        " str s0, [x13], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x14]                       \n\t"
        " fadd s0,s30,s0                      \n\t"
        " str s0, [x14], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x15]                       \n\t"
        " fadd s0,s31,s0                      \n\t"
        " str s0, [x15], #4                   \n\t"
        "                                     \n\t"
        " subs x16, x16, #1                   \n\t"
        " b.ne 1b                             \n\t"
        "                                     \n\t" 
        " 0:                                  \n\t" // Now the real parts, same code except for imag_in.
        "                                     \n\t"
        "                                     \n\t"
        " mov x16, %[m]                       \n\t" // Initialise loop variable.
        "                                     \n\t"
        "                                     \n\t" // No need to load addresses of imaginary parts.
        "                                     \n\t"
        " cmp x16, #4                         \n\t" // The following loop copies 4 elements per iteration
        " b.lo 1f                             \n\t" // Skip if we have less than four elements to copy.
        "                                     \n\t"
        " .align 5                            \n\t"
        " 2:                                  \n\t"
        FLD4inc(v16,v17,v18,v19,0,%[imag_in])
        FLD4inc(v20,v21,v22,v23,0,%[imag_in])
        FLD4inc(v24,v25,v26,v27,0,%[imag_in])
        FLD4inc(v28,v29,v30,v31,0,%[imag_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,1,%[imag_in])
        FLD4inc(v20,v21,v22,v23,1,%[imag_in])
        FLD4inc(v24,v25,v26,v27,1,%[imag_in])
        FLD4inc(v28,v29,v30,v31,1,%[imag_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,2,%[imag_in])
        FLD4inc(v20,v21,v22,v23,2,%[imag_in])
        FLD4inc(v24,v25,v26,v27,2,%[imag_in])
        FLD4inc(v28,v29,v30,v31,2,%[imag_in])
        "                                     \n\t"
        FLD4inc(v16,v17,v18,v19,3,%[imag_in])
        FLD4inc(v20,v21,v22,v23,3,%[imag_in])
        FLD4inc(v24,v25,v26,v27,3,%[imag_in])
        FLD4inc(v28,v29,v30,v31,3,%[imag_in])
        "                                     \n\t"
        " ldr q0, [x0]                        \n\t"
        FFADD(v0,v16,v0)
        " str q0, [x0], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x1]                        \n\t"
        FFADD(v0,v17,v0)
        " str q0, [x1], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x2]                        \n\t"
        FFADD(v0,v18,v0)
        " str q0, [x2], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x3]                        \n\t"
        FFADD(v0,v19,v0)
        " str q0, [x3], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x4]                        \n\t"
        FFADD(v0,v20,v0)
        " str q0, [x4], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x5]                        \n\t"
        FFADD(v0,v21,v0)
        " str q0, [x5], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x6]                        \n\t"
        FFADD(v0,v22,v0)
        " str q0, [x6], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x7]                        \n\t"
        FFADD(v0,v23,v0)
        " str q0, [x7], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x8]                        \n\t"
        FFADD(v0,v24,v0)
        " str q0, [x8], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x9]                        \n\t"
        FFADD(v0,v25,v0)
        " str q0, [x9], #16                   \n\t"
        "                                     \n\t"
        " ldr q0, [x10]                       \n\t"
        FFADD(v0,v26,v0)
        " str q0, [x10], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x11]                       \n\t"
        FFADD(v0,v27,v0)
        " str q0, [x11], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x12]                       \n\t"
        FFADD(v0,v28,v0)
        " str q0, [x12], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x13]                       \n\t"
        FFADD(v0,v29,v0)
        " str q0, [x13], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x14]                       \n\t"
        FFADD(v0,v30,v0)
        " str q0, [x14], #16                  \n\t"
        "                                     \n\t"
        " ldr q0, [x15]                       \n\t"
        FFADD(v0,v31,v0)
        " str q0, [x15], #16                  \n\t"
        "                                     \n\t"
        " sub x16, x16, #4                    \n\t"
        " cmp x16, #4                         \n\t"
        " b.hs 2b                             \n\t"
        "                                     \n\t"
        " cbz x16, 0f                         \n\t"
        " 1:                                  \n\t"    
        " ldr s16, [%[imag_in]], #4           \n\t"
        " ldr s17, [%[imag_in]], #4           \n\t"
        " ldr s18, [%[imag_in]], #4           \n\t"
        " ldr s19, [%[imag_in]], #4           \n\t"
        " ldr s20, [%[imag_in]], #4           \n\t"
        " ldr s21, [%[imag_in]], #4           \n\t"
        " ldr s22, [%[imag_in]], #4           \n\t"
        " ldr s23, [%[imag_in]], #4           \n\t"
        " ldr s24, [%[imag_in]], #4           \n\t"
        " ldr s25, [%[imag_in]], #4           \n\t"
        " ldr s26, [%[imag_in]], #4           \n\t"
        " ldr s27, [%[imag_in]], #4           \n\t"
        " ldr s28, [%[imag_in]], #4           \n\t"
        " ldr s29, [%[imag_in]], #4           \n\t"
        " ldr s30, [%[imag_in]], #4           \n\t"
        " ldr s31, [%[imag_in]], #4           \n\t"
        "                                     \n\t"
        " ldr s0, [x0]                        \n\t"
        " fadd s0,s16,s0                      \n\t"
        " str s0, [x0], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x1]                        \n\t"
        " fadd s0,s17,s0                      \n\t"
        " str s0, [x1], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x2]                        \n\t"
        " fadd s0,s18,s0                      \n\t"
        " str s0, [x2], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x3]                        \n\t"
        " fadd s0,s19,s0                      \n\t"
        " str s0, [x3], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x4]                        \n\t"
        " fadd s0,s20,s0                      \n\t"
        " str s0, [x4], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x5]                        \n\t"
        " fadd s0,s21,s0                      \n\t"
        " str s0, [x5], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x6]                        \n\t"
        " fadd s0,s22,s0                      \n\t"
        " str s0, [x6], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x7]                        \n\t"
        " fadd s0,s23,s0                      \n\t"
        " str s0, [x7], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x8]                        \n\t"
        " fadd s0,s24,s0                      \n\t"
        " str s0, [x8], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x9]                        \n\t"
        " fadd s0,s25,s0                      \n\t"
        " str s0, [x9], #4                    \n\t"
        "                                     \n\t"
        " ldr s0, [x10]                       \n\t"
        " fadd s0,s26,s0                      \n\t"
        " str s0, [x10], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x11]                       \n\t"
        " fadd s0,s27,s0                      \n\t"
        " str s0, [x11], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x12]                       \n\t"
        " fadd s0,s28,s0                      \n\t"
        " str s0, [x12], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x13]                       \n\t"
        " fadd s0,s29,s0                      \n\t"
        " str s0, [x13], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x14]                       \n\t"
        " fadd s0,s30,s0                      \n\t"
        " str s0, [x14], #4                   \n\t"
        "                                     \n\t"
        " ldr s0, [x15]                       \n\t"
        " fadd s0,s31,s0                      \n\t"
        " str s0, [x15], #4                   \n\t"
        "                                     \n\t"
        " subs x16, x16, #1                   \n\t"
        " b.ne 1b                             \n\t"
        "                                     \n\t"
        " 0:                                  \n\t"

        : // Output
            [solids]  "+m"(solids),
            [real_in] "+r"(real_in),
            [imag_in] "+r"(imag_in)

        : // Input
            [m]        "r"(m)

        : // Clobbered registers
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
        "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
        "x16", "v0",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
}

}

#endif

