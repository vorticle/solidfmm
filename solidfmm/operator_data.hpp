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
#ifndef SOLIDFMM_OPERATOR_DATA_HPP
#define SOLIDFMM_OPERATOR_DATA_HPP

#include <cstddef> // Needed for size_t.

namespace solidfmm
{

template <typename real> class microkernel;

template <typename real>
class operator_data
{
public:
    operator_data() = delete;
    operator_data( size_t P );
    operator_data( const operator_data&  rhs ) = delete;
    operator_data(       operator_data&& rhs ) = delete;
    operator_data& operator=( const operator_data  &rhs ) = delete;
    operator_data& operator=(       operator_data &&rhs ) = delete;
    ~operator_data();

    size_t order() const noexcept { return P; }
    const microkernel<real>* kernel() const noexcept { return kernel_; }

    const real* faculties() const noexcept { return buf_; }
    const real* inverse_faculties_decreasing( size_t n ) const noexcept;
    const real* inverse_faculties_increasing( size_t n ) const noexcept;

    const real* real_swap_matrix           ( size_t n ) const noexcept { return real_swap_mats   [n]; }
    const real* imag_swap_matrix           ( size_t n ) const noexcept { return imag_swap_mats   [n]; }
    const real* real_swap_matrix_transposed( size_t n ) const noexcept { return real_swap_mats_tr[n]; }
    const real* imag_swap_matrix_transposed( size_t n ) const noexcept { return imag_swap_mats_tr[n]; }
    
private:
    size_t P;
    real  *buf_;
    real **ptrbuf_;
    const microkernel<real>* kernel_;

    real *invfac_decreasing;
    real *invfac_increasing;

    real** real_swap_mats;
    real** imag_swap_mats;
    real** real_swap_mats_tr;
    real** imag_swap_mats_tr;
};

extern template class operator_data<float>;
extern template class operator_data<double>;

}

#endif

