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
#ifndef SOLIDFMM_MICROKERNEL_HPP
#define SOLIDFMM_MICROKERNEL_HPP

#include <cstddef> // Needed for size_t

namespace solidfmm
{

template <typename real>
class microkernel
{
public:
    const size_t rows;
    const size_t cols;
    const size_t alignment;

    microkernel( size_t p_rows, size_t p_cols, size_t p_alignment ) noexcept:
    rows { p_rows }, cols { p_cols }, alignment { p_alignment } {}

    virtual ~microkernel() = default;


    virtual void euler( const real *x,   const real *y, const real *z,
                              real *r,         real *rinv,
                              real *cos_alpha, real *sin_alpha,
                              real *cos_beta,  real *sin_beta,
                        size_t k ) const noexcept = 0;

    virtual void rotscale( const real *cos,      const real *sin,      const real *scale,
                           const real *real_in,  const real *imag_in,
                                 real *real_out,       real *imag_out,
                           size_t k, bool forward ) const noexcept = 0;

    virtual void swap( const real *mat, const real *in,
                       real *out, size_t k, bool pattern ) const noexcept = 0;

    virtual void zm2l( const real *mat, const real *in, 
                       real *out, size_t k, bool pattern ) const noexcept = 0;

    virtual void zm2m( const real *mat, const real *in,
                       real *out, size_t k ) const noexcept = 0;

    
    virtual void swap2trans_buf( const real  *real_in,  const real  *imag_in,
                                       real **real_out,       real **imag_out,
                                 size_t n ) const noexcept = 0;

    virtual void trans2swap_buf( const real *const *const real_in, 
                                 const real *const *const imag_in,
                                       real *real_out, real *imag_out,
                                 size_t n, size_t Pmax ) const noexcept = 0;

    virtual void solid2buf( const real *const *solids, const size_t *P,
                            real *real_out, real *imag_out, size_t n ) const noexcept = 0;

    virtual void buf2solid( const real *real_in, const real *imag_in, 
                            real **solids, const size_t *P, size_t n ) const noexcept = 0;
};

extern template class microkernel<float >;
extern template class microkernel<double>;

template <typename real> microkernel<real>* get_microkernel();
template <> microkernel<float >* get_microkernel<float >();
template <> microkernel<double>* get_microkernel<double>();

}

#endif

