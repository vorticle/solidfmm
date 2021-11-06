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
#ifndef SOLIDFMM_TRHEADLOCAL_BUFFER_HPP
#define SOLIDFMM_TRHEADLOCAL_BUFFER_HPP

#include <cstddef> // Needed for size_t.

namespace solidfmm
{

template <typename real> class operator_data;

template <typename real>
class threadlocal_buffer
{
public:
    threadlocal_buffer() noexcept;
    threadlocal_buffer( const operator_data<real> &op );
    threadlocal_buffer( const threadlocal_buffer  &rhs );
    threadlocal_buffer(       threadlocal_buffer &&rhs ) noexcept;
    threadlocal_buffer& operator=( const threadlocal_buffer  &rhs );
    threadlocal_buffer& operator=(       threadlocal_buffer &&rhs ) noexcept;
   ~threadlocal_buffer();

    void swap( threadlocal_buffer &rhs ) noexcept;

    void initialise( const operator_data<real> &op );
    bool compatible( const operator_data<real> &op ) const noexcept;

private:
    size_t P, rows, cols;
    size_t alignment, bufsize;
    real *buf, **ptr_buf;
    size_t *Pbuf;

public:
    real *r, *rinv;
    real *cos_alpha, *sin_alpha;
    real *cos_beta,  *sin_beta;

    real   *swap_real_in,    *swap_imag_in; 
    real   *swap_real_out,   *swap_imag_out; 
    real **trans_real_in,  **trans_imag_in;
    real **trans_real_out, **trans_imag_out;

    real **solid_in, **solid_out;
    size_t  *P_in, *P_out;
    real    *x, *y, *z;
};

extern template class threadlocal_buffer<float>;
extern template class threadlocal_buffer<double>;

}

#endif

