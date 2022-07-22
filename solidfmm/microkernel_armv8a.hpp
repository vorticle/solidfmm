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
#ifndef SOLIDFMM_MICROKERNEL_ARMV8A_HPP
#define SOLIDFMM_MICROKERNEL_ARMV8A_HPP

#if __ARM_ARCH >= 8 && __ARM_ARCH < 900 && __ARM_ARCH_PROFILE == 'A' && __ARM_ARCH_ISA_A64 
#define SOLIDFMM_ARMv8A 1
#endif

#ifdef SOLIDFMM_ARMv8A 
#include <solidfmm/microkernel.hpp>


namespace solidfmm
{

class microkernel_float_armv8a: public microkernel<float>
{
public:
    microkernel_float_armv8a();

    virtual ~microkernel_float_armv8a() = default;

    virtual void euler( const float *x,   const float *y, const float *z,
                              float *r,         float *rinv,
                              float *cos_alpha, float *sin_alpha,
                              float *cos_beta,  float *sin_beta,
                       size_t k ) const noexcept override;

    virtual void rotscale( const float *cos,      const float *sin,      const float *scale,
                           const float *real_in,  const float *imag_in,
                                 float *real_out,       float *imag_out,
                           size_t k, bool forward ) const noexcept override;

    virtual void swap( const float *mat, const float *in,
                             float *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2l( const float *mat, const float *in, 
                             float *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2m( const float *mat, const float *in, 
                             float *out, size_t k ) const noexcept override;

    virtual void swap2trans_buf( const float  *real_in,  const float  *imag_in,
                                       float **real_out,       float **imag_out,
                                 size_t n ) const noexcept override;

    virtual void trans2swap_buf( const float *const *const real_in, 
                                 const float *const *const imag_in,
                                       float *real_out, float *imag_out,
                                 size_t n, size_t Pmax ) const noexcept override;

    virtual void solid2buf( const float *const *solids, const float *const zeros, const size_t *P,
                                  float *real_out, float *imag_out,
                            size_t n ) const noexcept override;

    virtual void buf2solid( const float  *real_in, const float *imag_in, 
                                  float **solids, float *trash, const size_t *P,
                            size_t n ) const noexcept override;
};

class microkernel_double_armv8a: public microkernel<double>
{
public:
    microkernel_double_armv8a();

    virtual ~microkernel_double_armv8a() = default;

    virtual void euler( const double *x,   const double *y, const double *z,
                              double *r,         double *rinv,
                              double *cos_alpha, double *sin_alpha,
                              double *cos_beta,  double *sin_beta,
                      size_t k ) const noexcept override;

    virtual void rotscale( const double *cos,      const double *sin,      const double *scale,
                           const double *real_in,  const double *imag_in,
                                 double *real_out,       double *imag_out,
                           size_t k, bool forward ) const noexcept override;

    virtual void swap( const double *mat, const double *in,
                             double *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2l( const double *mat, const double *in, 
                             double *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2m( const double *mat, const double *in, 
                             double *out, size_t k ) const noexcept override;

    virtual void swap2trans_buf( const double  *real_in,  const double  *imag_in,
                                       double **real_out,       double **imag_out,
                                size_t n ) const noexcept override;

    virtual void trans2swap_buf( const double *const *const real_in, 
                                 const double *const *const imag_in,
                                       double *real_out, double *imag_out,
                                 size_t n, size_t Pmax ) const noexcept override;

    virtual void solid2buf( const double *const *solids, const double *const zeros, const size_t *P,
                                  double *real_out, double *imag_out, 
                            size_t n ) const noexcept override;

    virtual void buf2solid( const double  *real_in, const double *imag_in, 
                                  double **solids,        double *trash, const size_t *P,
                            size_t n ) const noexcept override;
};

}

#endif
#endif

