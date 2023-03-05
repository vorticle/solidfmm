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
#include <solidfmm/microkernel_test.hpp>

#include <cmath>     // std::hypot, std::sqrt
#include <algorithm> // For std::min

namespace solidfmm
{

template <typename real>
void 
microkernel_test<real>::euler( const real *x,   const real *y, const real *z,
                                     real *r,         real *rinv,
                                     real *cos_alpha, real *sin_alpha,
                                     real *cos_beta,  real *sin_beta,
                                     size_t k ) const noexcept
{
    const size_t cols = this->cols;

    if ( k )
    {
        for ( size_t l = 0; l < cols; ++l )
        {
            r        [ l ] = 1;
            rinv     [ l ] = 1;
            cos_alpha[ l ] = 1;
            sin_alpha[ l ] = 0;
            cos_beta [ l ] = 1;
            sin_beta [ l ] = 0;
        }
    }

    if ( k > 1 )
    {
        for ( size_t l = 0; l < cols; ++l )
        {
            real rxy  = std::hypot(x[l],y[l]);
            real rxyz = std::hypot(rxy ,z[l]);
            r        [ cols + l ] = rxyz;
            rinv     [ cols + l ] = 1/rxyz;
            cos_alpha[ cols + l ] = (rxy == 0) ? 1 : y[l]/rxy;
            sin_alpha[ cols + l ] = (rxy == 0) ? 0 : x[l]/rxy;
            cos_beta [ cols + l ] =  z[l]*rinv[ cols + l ];
            sin_beta [ cols + l ] =  -rxy*rinv[ cols + l ];
        }
    }

    for ( size_t i = 2; i < k;    ++i )
    for ( size_t l = 0; l < cols; ++l )
    {
        r   [ i*cols + l ] = r   [ cols + l ]*r   [ (i-1)*cols + l ];
        rinv[ i*cols + l ] = rinv[ cols + l ]*rinv[ (i-1)*cols + l ];

        
        real c  = cos_alpha[ cols + l ];
        real s  = sin_alpha[ cols + l ];
        real cn = cos_alpha[ (i-1)*cols + l ]; 
        real sn = sin_alpha[ (i-1)*cols + l ]; 
        cos_alpha[ i*cols + l ] = cn*c - sn*s;
        sin_alpha[ i*cols + l ] = sn*c + cn*s;

        c  = cos_beta[ cols + l ];
        s  = sin_beta[ cols + l ];
        cn = cos_beta[ (i-1)*cols + l ]; 
        sn = sin_beta[ (i-1)*cols + l ]; 
        cos_beta[ i*cols + l ] = cn*c - sn*s;
        sin_beta[ i*cols + l ] = sn*c + cn*s;
    }
}

template <typename real>
void microkernel_test<real>::
rotscale( const real *cos, const real *sin,      const real *scale,
                           const real *real_in,  const real *imag_in,
                                 real *real_out,       real *imag_out,
                             size_t k, bool forward ) const noexcept
{
    const size_t cols = this->cols;

    if ( forward )
    {
        for ( size_t i = 0; i < k;    ++i )
        for ( size_t l = 0; l < cols; ++l )
        {
            real c   = cos    [ i*cols + l ];
            real s   = sin    [ i*cols + l ];
            real re  = real_in[ i*cols + l ];
            real im  = imag_in[ i*cols + l ];
            real fac = scale  [          l ];
        
            real_out[ i*cols + l ] = fac*(c*re - s*im);
            imag_out[ i*cols + l ] = fac*(s*re + c*im);
        }
    }
    else
    {
        for ( size_t i = 0; i < k;    ++i )
        for ( size_t l = 0; l < cols; ++l )
        {
            real c   = cos    [ i*cols + l ];
            real s   = sin    [ i*cols + l ];
            real re  = real_in[ i*cols + l ];
            real im  = imag_in[ i*cols + l ];
            real fac = scale  [          l ];
        
            real_out[ i*cols + l ] = fac*( c*re + s*im);
            imag_out[ i*cols + l ] = fac*(-s*re + c*im);
        }
    }
}

template <typename real>
void microkernel_test<real>::
swap( const real *mat, const real *in,
                             real *out, size_t k, bool pattern ) const noexcept
{
    const size_t rows = this->rows;
    const size_t cols = this->cols;

    // Dynamic memory allocation is painfully slow! In you own microkernel
    // the values of "rows" and "cols" should be known at compile time and you
    // should allocate the result matrix statically or hold it in registers.
    // Also, this might throw and break the noexcept behaviour.
    real *c = new real[ rows * cols ] {};
    bool k_odd = k &  1;
         k     = k >> 1;

    while ( k-- )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i*cols+j] = c[i*cols+j] + fac*in[j];
        }
        in = in + cols;
        
        for ( size_t i = 1; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i*cols+j] = c[i*cols+j] + fac*in[j];
        }
        in  = in + cols;
    }

    if ( k_odd )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i*cols+j] = c[i*cols+j] + fac*in[j];
        }
        in = in + cols;
    }

    if ( pattern )
    {
        for ( size_t i = 0; i < rows/2; ++i )
        {
            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[ (2*i + 0)*cols + j ];

            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[ (2*i + 1)*cols + j ];
        }
    }
    else
    {
        for ( size_t i = 0; i < rows/2; ++i )
        {
            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[(2*i + 1)*cols + j ];

            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[(2*i + 0)*cols + j ];
        }
    }

    delete [] c;
}

template <typename real>
void microkernel_test<real>::
zm2l( const real *mat, const real *in, 
            real *out, size_t k, bool pattern ) const noexcept
{
    const size_t rows = this->rows;
    const size_t cols = this->cols;

    // Dynamic memory allocation is painfully slow! In you own microkernel
    // the values of "rows" and "cols" should be known at compile time and you
    // should allocate the result matrix statically or hold it in registers.
    // Also, this might throw and break the noexcept behaviour.
    real *c = new real[rows*cols] {};

    while ( k-- )
    {
        const real *faculty_column = mat;
        for ( size_t i = 0; i < rows; ++i )
        {
            const real f = *faculty_column++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i*cols+j] = c[i*cols+j] + f*in[j];
        }

        in  = in  + cols;
        mat = mat + 1;
    }

    if ( pattern )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        for ( size_t j = 0; j < cols; ++j  )
            c[i*cols+j] = -c[i*cols+j];
    }
    else
    {
        for ( size_t i = 1; i < rows; i += 2 )
        for ( size_t j = 0; j < cols; ++j  )
            c[i*cols+j] = -c[i*cols+j];
    }

    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
        *out++ = c[i*cols+j];

    delete [] c;
}

template <typename real>
void microkernel_test<real>::
zm2m( const real *mat, const real *in,
            real *out, size_t k ) const noexcept
{
    const size_t rows = this->rows;
    const size_t cols = this->cols;

    // Dynamic memory allocation is painfully slow! In you own microkernel
    // the values of "rows" and "cols" should be known at compile time and you
    // should allocate the result matrix statically or hold it in registers.
    // Also, this might throw and break the noexcept behaviour.
    real *c = new real[ rows*cols ] {};

    while ( k-- )
    {
        const real *faculty_column = mat;
        for ( size_t i = 0; i < rows; ++i )
        {
            const real f = *faculty_column++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i*cols+j] = c[i*cols+j] + f*in[j];
        }

        in  = in  + cols;
        mat = mat + 1;
    }

    // Store the rows in reverse order.
    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
        *out++ = c[(rows-1-i)*cols+j];

    delete [] c;
}

template <typename real>
void microkernel_test<real>::
swap2trans_buf( const real  *real_in,  const real  *imag_in,
                      real **real_out,       real **imag_out,
                      size_t n ) const noexcept 
{
    const size_t cols = this->cols;

    for ( size_t m = 0; m <= n;    ++m )
    for ( size_t k = 0; k <  cols; ++k )
    {
        real_out[m][ (n-m)*cols + k ] = real_in[ m*cols + k ];
        imag_out[m][ (n-m)*cols + k ] = imag_in[ m*cols + k ];
    }
}

template <typename real>
void microkernel_test<real>::
trans2swap_buf( const real *const *const real_in, 
                const real *const *const imag_in,
                      real *real_out, real *imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    using std::min;
    const size_t cols = this->cols;

    // First the entries where m < Pmax
    for ( size_t m = 0; m < min(Pmax,n+1); ++m )
    for ( size_t k = 0; k < cols;          ++k )
    {
        real_out[ m*cols + k ] = real_in[m][ (n-m)*cols + k ];
        imag_out[ m*cols + k ] = imag_in[m][ (n-m)*cols + k ];
    }

    // Now the remaining entries, if any.
    for ( size_t m = Pmax; m <= n;    ++m )
    for ( size_t k = 0;    k <  cols; ++k )
    {
        real_out[ m*cols + k ] = 0;
        imag_out[ m*cols + k ] = 0;
    }
}

template <typename real>
void microkernel_test<real>::
solid2buf( const real *const *p_solids, const real *const zeros, const size_t *P,
                 real *real_out, real *imag_out, size_t n ) const noexcept
{
    const size_t cols = this->cols;
    real** solids = new real*[ cols ];

    for ( size_t l = 0; l < cols; ++l )
    {
        if ( n < P[l] )
            solids[ l ] = const_cast<real*>(p_solids[l]);
        else
            solids[ l ] = const_cast<real*>(zeros);
    }

    for ( size_t l = 0; l < cols; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        real_out[ m*cols + l ] = solids[l][ (n  )*(n+1) + m ];
        imag_out[ m*cols + l ] = solids[l][ (n+1)*(n+1) + m ];
    }

    delete [] solids;
}

template <typename real>
void microkernel_test<real>::
buf2solid( const real *real_in, const real *imag_in, 
                 real **p_solids, real *trash, const size_t *P, size_t n ) const noexcept
{
    const size_t cols = this->cols;
    real** solids = new real*[ cols ];

    for ( size_t l = 0; l < cols; ++l )
    {
        if ( n < P[l] )
            solids[ l ] = p_solids[l];
        else
            solids[ l ] = trash;
    }

    for ( size_t l = 0; l < cols; ++l )
    for ( size_t m = 0; m <= n; ++m )
    {
        solids[l][ (n  )*(n+1) + m ] += real_in[ m*cols + l ];
        solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*cols + l ];
    } 

    delete [] solids;
}

// Explicit instantiations.
template class microkernel_test<float >;
template class microkernel_test<double>;

}

