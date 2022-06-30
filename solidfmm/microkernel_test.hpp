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
#ifndef SOLIDFMM_MICROKERNEL_TEST_HPP
#define SOLIDFMM_MICROKERNEL_TEST_HPP

#include <solidfmm/microkernel.hpp>

namespace solidfmm
{

// This class is used for testing new microkernels only.
// WARNING: NEVER USE THIS IN PRODUCTION CODE, IT IS NOT EXCEPTION SAFE!

template <typename real>
class microkernel_test: public microkernel<real>
{
public:
    microkernel_test( size_t p_rows, size_t p_cols, size_t p_alignment ) noexcept:
    microkernel<real>( p_rows, p_cols, p_alignment ) {}

    virtual ~microkernel_test() = default;

    virtual void euler( const real *x,   const real *y, const real *z,
                              real *r,         real *rinv,
                              real *cos_alpha, real *sin_alpha,
                              real *cos_beta,  real *sin_beta,
                        size_t k ) const noexcept override;

    virtual void rotscale( const real *cos,      const real *sin,      const real *scale,
                           const real *real_in,  const real *imag_in,
                                 real *real_out,       real *imag_out,
                           size_t k, bool forward ) const noexcept override;

    virtual void swap( const real *mat, const real *in,
                       real *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2l( const real *mat, const real *in, 
                       real *out, size_t k, bool pattern ) const noexcept override;

    virtual void zm2m( const real *mat, const real *in,
                       real *out, size_t k ) const noexcept override;

    
    virtual void swap2trans_buf( const real  *real_in,  const real  *imag_in,
                                       real **real_out,       real **imag_out,
                                 size_t n ) const noexcept override;

    virtual void trans2swap_buf( const real *const *const real_in, 
                                 const real *const *const imag_in,
                                       real *real_out, real *imag_out,
                                 size_t n, size_t Pmax ) const noexcept override;

    virtual void solid2buf( const real *const *solids, const real *const zeros, const size_t *P,
                            real *real_out, real *imag_out, size_t n ) const noexcept override;

    virtual void buf2solid( const real *real_in, const real *imag_in, 
                            real **solids, real *trash, const size_t *P, size_t n ) const noexcept override;
};

extern template class microkernel_test<float >;
extern template class microkernel_test<double>;

}

#endif

