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
#include <solidfmm/operator_data.hpp>
#include <solidfmm/swap_matrix.hpp>
#include <solidfmm/microkernel.hpp>

#include <memory>
#include <stdexcept>

#include <iostream>

namespace solidfmm
{

namespace
{

template <typename real> constexpr size_t Pmax();

template <> constexpr size_t Pmax<float>()
{
    return 18;
}

template <> constexpr size_t Pmax<double>()
{
    return 86;
}

template <typename real>
real* comp_realmat_tr( size_t n, real *target, size_t rows, const swap_matrix<real> &B ) noexcept
{
    // Pattern:
    // If true, then the very first entry of the matrix is non-zero.
    bool pattern = !(n&1);
    for ( int r = 0; r <= int(n)   ; r += int(rows) )
    for ( int l = 0; l <= int(n)   ; ++l )
    for ( int i = 0; i <  int(rows); ++i )
    {
        int m = r + i;
        bool case1 =  pattern && !((l+m)&1);
        bool case2 = !pattern &&  ((l+m)&1);
        if ( case1 || case2 ) 
        {
             real v;
                  if ( m > int(n) ) v = 0;
             else if ( l == 0 )     v = B(n,m,0);
             else if ( l &  1 )     v = B(n,m,l) - B(n,m,-l);
             else                   v = B(n,m,l) + B(n,m,-l);
            *target++ = v;
        }
    }
    return target;
}

template <typename real>
real* comp_imagmat_tr( size_t n, real *target, size_t rows, const swap_matrix<real> &B ) noexcept
{
    // Pattern:
    // If true, then the very first entry of the matrix is non-zero.
    bool pattern = (n&1);
    for ( int r = 0; r <= int(n)   ; r += int(rows) )
    for ( int l = 0; l <= int(n)   ; ++l )
    for ( int i = 0; i <  int(rows); ++i )
    {
        int m = r + i;
        bool case1 =  pattern && !((l+m)&1);
        bool case2 = !pattern &&  ((l+m)&1);
        if ( case1 || case2 ) 
        {
             real v;
                  if ( m > int(n) ) v = 0;
             else if ( l &  1 )     v = B(n,m,l) + B(n,m,-l);
             else                   v = B(n,m,l) - B(n,m,-l);
            *target++ = v;
        }
    }
    return target;
}

template <typename real>
real* comp_realmat( size_t n, real *target, size_t rows, const swap_matrix<real> &B ) noexcept
{
    // Pattern:
    // If true, then the very first entry of the matrix is non-zero.
    bool pattern = !(n&1);
    for ( int r = 0; r <= int(n)   ; r += int(rows) )
    for ( int l = 0; l <= int(n)   ; ++l )
    for ( int i = 0; i <  int(rows); ++i )
    {
        int m = r + i;
        bool case1 =  pattern && !((l+m)&1);
        bool case2 = !pattern &&  ((l+m)&1);
        if ( case1 || case2 ) 
        {
             real v;
                  if ( m > int(n) ) v = 0;
             else if ( l == 0 )     v = B(n,0,m);
             else if ( l &  1 )     v = B(n,l,m) - B(n,-l,m);
             else                   v = B(n,l,m) + B(n,-l,m);
            *target++ = v;
        }
    }
    return target;
}

template <typename real>
real* comp_imagmat( size_t n, real *target, size_t rows, const swap_matrix<real> &B ) noexcept
{
    // Pattern:
    // If true, then the very first entry of the matrix is non-zero.
    bool pattern = (n&1);
    for ( int r = 0; r <= int(n)   ; r += int(rows) )
    for ( int l = 0; l <= int(n)   ; ++l )
    for ( int i = 0; i <  int(rows); ++i )
    {
        int m = r + i;
        bool case1 =  pattern && !((l+m)&1);
        bool case2 = !pattern &&  ((l+m)&1);
        if ( case1 || case2 ) 
        {
             real v;
                  if ( m >  int(n) ) v = 0;
             else if ( l &  1 )      v = B(n,l,m) + B(n,-l,m);
             else                    v = B(n,l,m) - B(n,-l,m);
            *target++ = v;
        }
    }
    return target;
}

}

template <typename real> operator_data<real>::operator_data( size_t arg_P ):
 P        { arg_P },
    buf_  { nullptr }, 
 ptrbuf_  { nullptr },
 kernel_  { nullptr }
{
    if ( P == 0 )
        throw std::logic_error { "solidfmm::operator_data(): "
                                 "Requested order P = 0." };

    if ( P > Pmax<real>() )
        throw std::overflow_error
        { "solidfmm::operator_data(): "
          "Requested order is too large for the chosen floating point type." };

    using kernel_t = const microkernel<real>;

    std::unique_ptr<kernel_t> kern   { get_microkernel<real>() };
    std::unique_ptr<real*[]>  ptrbuf { new real*[ 4*P ] };

    // How much memory do we need?
    // 2*P - 1 for storing all faculties 0!, 1!, 2!, ..., (2*(P-1))!
    // kernel_rows - 1 additional padding.
    // P Real Swap matrices
    // P Imag Swap matrices
    // P Real Swap matrices transposed
    // P Imag Swap matrices transposed
    const size_t kernel_rows = kern->rows;
    size_t num_faculties = 2*P - 1 + kernel_rows - 1;

    size_t bufsize = num_faculties;

    size_t num_invfaculties = P + kernel_rows - 1;
    bufsize += 2*num_invfaculties;

    for ( size_t n = 0; n < P; ++n )
    {
        size_t row_chunks = 1 + n/kernel_rows;
        size_t rows = row_chunks * kernel_rows / 2;
        size_t cols = n+1;
        bufsize = bufsize + 4*rows*cols;
    }
    std::unique_ptr<real[]> buf { new real[bufsize] };

    buf[ 0 ] = static_cast<real>(1);
    for ( size_t n = 1; n < 2*P - 1; ++n )
        buf[n] = static_cast<real>(n) * buf[n-1];

    for ( size_t n = 2*P-1; n < num_faculties; ++n )
        buf[n] = 0;

    invfac_decreasing = buf.get() + num_faculties;
    for ( size_t n = 0; n < P; ++n )
        invfac_decreasing[n] = 1/buf[P-1-n];

    for ( size_t n = P; n < num_invfaculties; ++n )
        invfac_decreasing[n] = 0;

    invfac_increasing = buf.get() + num_faculties + num_invfaculties;
    for ( size_t i = 0; i < kernel_rows-1; ++i )
        invfac_increasing[i] = 0;

    for ( size_t i = 0; i < P; ++i )
        invfac_increasing[i+kernel_rows-1] = 1/buf[i];


    swap_matrix<real> B(P);
    real *current = buf.get() + num_faculties + 2*num_invfaculties;
    real_swap_mats    = ptrbuf.get() + 0*P;
    imag_swap_mats    = ptrbuf.get() + 1*P;
    real_swap_mats_tr = ptrbuf.get() + 2*P;
    imag_swap_mats_tr = ptrbuf.get() + 3*P;
    for ( size_t n = 0; n < P; ++n )
    {
        real_swap_mats[n] = current;
        current = comp_realmat( n, current, kern->rows, B );

        imag_swap_mats[n] = current;
        current = comp_imagmat( n, current, kern->rows, B );
    }

    for ( size_t n = 0; n < P; ++n )
    {
        real_swap_mats_tr[n] = current;
        current = comp_realmat_tr( n, current, kern->rows, B );

        imag_swap_mats_tr[n] = current;
        current = comp_imagmat_tr( n, current, kern->rows, B );
    }

    kernel_ =   kern.release();
    ptrbuf_ = ptrbuf.release();
       buf_ =    buf.release();
}

template <typename real>
operator_data<real>::~operator_data()
{
    delete []    buf_;
    delete [] ptrbuf_;
    delete    kernel_;
}

template <typename real>
const real* operator_data<real>::inverse_faculties_increasing( size_t n ) const noexcept
{
    return invfac_increasing + kernel_->rows - 1 + n;
}

template <typename real>
const real* operator_data<real>::inverse_faculties_decreasing( size_t n ) const noexcept
{
    return invfac_decreasing + (P-1) - n;
}

//////////////////////////////////////////////////////////////
// Explicit instantiations for single and double precision. //
//////////////////////////////////////////////////////////////

template class operator_data<float>;
template class operator_data<double>;

}

