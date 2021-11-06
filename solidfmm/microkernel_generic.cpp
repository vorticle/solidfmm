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
#include <solidfmm/microkernel_generic.hpp>

#include <cmath>

#pragma GCC optimize("O2")
#pragma GCC optimize("tree-vectorize")

namespace solidfmm
{

namespace
{

template <typename real, size_t cols> inline
void euler_impl( const real *__restrict__ x,   
                 const real *__restrict__ y,
                 const real *__restrict__ z,
                       real *__restrict__ r,
                       real *__restrict__ rinv,
                       real *__restrict__ cos_alpha, 
                       real *__restrict__ sin_alpha,
                       real *__restrict__ cos_beta,  
                       real *__restrict__ sin_beta,
                       size_t k ) noexcept
{
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

template <typename real, size_t cols> inline
void rotscale_impl( const real *__restrict__ Cos,
                    const real *__restrict__ Sin,
                    const real *__restrict__ scale,
                    const real *__restrict__ real_in,
                    const real *__restrict__ imag_in,
                          real *__restrict__ real_out,
                          real *__restrict__ imag_out,
                    size_t k, bool forward ) noexcept
{

    if ( forward )
    {
        for ( size_t i = 0; i < k;    ++i )
        for ( size_t l = 0; l < cols; ++l )
        {
            real c   = Cos    [ i*cols + l ];
            real s   = Sin    [ i*cols + l ];
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
            real c   = Cos    [ i*cols + l ];
            real s   = Sin    [ i*cols + l ];
            real re  = real_in[ i*cols + l ];
            real im  = imag_in[ i*cols + l ];
            real fac = scale  [          l ];
        
            real_out[ i*cols + l ] = fac*( c*re + s*im);
            imag_out[ i*cols + l ] = fac*(-s*re + c*im);
        }
    }
}

template <typename real, size_t rows, size_t cols> inline
void swap_impl( const real *__restrict__ mat,
                const real *__restrict__ in,
                      real *__restrict__ out,
                      size_t k, bool pattern ) noexcept
{
    real c[rows][cols] {};
    bool k_odd = k &  1;
         k     = k >> 1;

    while ( k-- )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i][j] = c[i][j] + fac*in[j];
        }
        in = in + cols;
        
        for ( size_t i = 1; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i][j] = c[i][j] + fac*in[j];
        }
        in  = in + cols;
    }

    if ( k_odd )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        {
            const real fac = *mat++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i][j] = c[i][j] + fac*in[j];
        }
        in = in + cols;
    }

    if ( pattern )
    {
        for ( size_t i = 0; i < rows/2; ++i )
        {
            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[2*i + 0][j];

            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[2*i + 1][j];
        }
    }
    else
    {
        for ( size_t i = 0; i < rows/2; ++i )
        {
            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[2*i + 1][j];

            for ( size_t j = 0; j < cols; ++j )
                *out++ = c[2*i + 0][j];
        }
    }
}

template <typename real, size_t rows, size_t cols> inline
void zm2l_impl( const real *__restrict__ fac,
                const real *__restrict__ in,
                      real *__restrict__ out,
                      size_t k, bool pattern ) noexcept
{
    real c[rows][cols] {};

    while ( k-- )
    {
        const real *faculty_column = fac;
        for ( size_t i = 0; i < rows; ++i )
        {
            const real f = *faculty_column++;
            for ( size_t j = 0; j < cols; ++j  )
                c[i][j] = c[i][j] + f*in[j];
        }

        in  = in  + cols;
        fac = fac + 1;
    }

    if ( pattern )
    {
        for ( size_t i = 0; i < rows; i += 2 )
        for ( size_t j = 0; j < cols; ++j  )
            c[i][j] = -c[i][j];
    }
    else
    {
        for ( size_t i = 1; i < rows; i += 2 )
        for ( size_t j = 0; j < cols; ++j  )
            c[i][j] = -c[i][j];
    }

    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
        *out++ = c[i][j];
}

template <typename real, size_t rows, size_t cols> inline
void zm2m_impl( const real *__restrict__ fac,
                const real *__restrict__ in,
                      real *__restrict__ out,
                      size_t k ) noexcept
{
    real c[rows][cols] {};

    while ( k-- )
    {
        const real *faculty_column = fac;
        for ( size_t i = 0; i < rows; ++i )
        {
            const real f = *faculty_column++;
            for ( size_t j = 0; j < cols; ++j )
                c[i][j] = c[i][j] + f*in[j];
        }

        in  = in  + cols;
        fac = fac + 1;
    }

    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
        *out++ = c[rows-1-i][j];
}


template <typename real, size_t cols>
void swap2trans_buf_impl( const real *__restrict__ real_in, 
                          const real *__restrict__ imag_in,
                                real **real_out, 
                                real **imag_out,
                          size_t n ) noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        real *__restrict__ rdst = real_out[m] + (n-m)*cols;
        real *__restrict__ idst = imag_out[m] + (n-m)*cols;
        for ( size_t l = 0; l < cols; ++l )
        {
            rdst[l] = real_in[l];
            idst[l] = imag_in[l];
        }
        real_in += cols;
        imag_in += cols;
    }
}

template <typename real, size_t cols> inline
void trans2swap_buf_impl( const real *const *const real_in, 
                          const real *const *const imag_in,
                                real *__restrict__ real_out, 
                                real *__restrict__ imag_out,
                                size_t n, size_t Pmax ) noexcept
{
    for ( size_t m = 0; m <= n; ++m )
    {
        if ( m < Pmax )
        {
            const real *__restrict__ rsrc = real_in[m] + (n-m)*cols;
            const real *__restrict__ isrc = imag_in[m] + (n-m)*cols;
            for ( size_t l = 0; l < cols; ++l )
            {
                real_out[l] = rsrc[l];
                imag_out[l] = isrc[l];
            }
        
        }
        else
        {
            for ( size_t l = 0; l < cols; ++l )
            {
                real_out[l] = 0;
                imag_out[l] = 0;
            }
        }
        real_out += cols;
        imag_out += cols;
    }
}

template <typename real, size_t cols>
void solid2buf_impl( const real *const *solids, const size_t *P,
                     real *real_out, real *imag_out, size_t n ) noexcept
{
    for ( size_t l = 0; l < cols; ++l )
    {
        if ( n < P[l] )
        {
            for ( size_t m = 0; m <= n; ++m )
            {
                real_out[ m*cols + l ] = solids[l][ (n  )*(n+1) + m ];
                imag_out[ m*cols + l ] = solids[l][ (n+1)*(n+1) + m ];
            }
        }
        else
        {
            for ( size_t m = 0; m <= n; ++m )
            {
                real_out[ m*cols + l ] = 0;
                imag_out[ m*cols + l ] = 0;
            }
        }
    }
}

template <typename real, size_t cols>
void buf2solid_impl( const real *real_in, const real *imag_in, 
                     real **solids, const size_t *P, size_t n ) noexcept
{
    for ( size_t l = 0; l < cols; ++l )
    {
        if ( n < P[l] )
        {
            for ( size_t m = 0; m <= n; ++m )
            {
                solids[l][ (n  )*(n+1) + m ] += real_in[ m*cols + l ];
                solids[l][ (n+1)*(n+1) + m ] += imag_in[ m*cols + l ];
            } 
        }
    }
}

}

void microkernel_float_generic::
euler( const float *x,   const float *y, const float *z,
             float *r,         float *rinv,
             float *cos_alpha, float *sin_alpha,
             float *cos_beta,  float *sin_beta,
             size_t k ) const noexcept
{
    euler_impl<float,8>(x,y,z,r,rinv,cos_alpha,sin_alpha,cos_beta,sin_beta, k );
}

void microkernel_float_generic::
rotscale( const float *cos,      const float *sin,      const float *scale,
          const float *real_in,  const float *imag_in,
                float *real_out,       float *imag_out,
                size_t k, bool forward ) const noexcept
{
    rotscale_impl<float,8>(cos,sin,scale,real_in,imag_in,real_out,imag_out,k,forward);
}


void microkernel_float_generic::
swap( const float *mat, const float *in,
            float *out, size_t k, bool pattern ) const noexcept
{
    swap_impl<float,4,8>(mat,in,out,k,pattern);
}

void microkernel_float_generic::
zm2l( const float *mat, const float *in,
            float *out, size_t k, bool pattern ) const noexcept
{
    zm2l_impl<float,4,8>(mat,in,out,k,pattern);
}

void microkernel_float_generic::
zm2m( const float *mat, const float *in,
            float *out, size_t k ) const noexcept
{
    zm2m_impl<float,4,8>(mat,in,out,k);
}

void microkernel_float_generic::
swap2trans_buf( const float *__restrict__ real_in, 
                const float *__restrict__ imag_in,
                      float **real_out, 
                      float **imag_out,
                      size_t n ) const noexcept
{
    swap2trans_buf_impl<float,8>(real_in,imag_in,real_out,imag_out,n);
}

void microkernel_float_generic::
trans2swap_buf( const float *const *const real_in, 
                const float *const *const imag_in,
                      float *__restrict__ real_out, 
                      float *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    trans2swap_buf_impl<float,8>(real_in,imag_in,real_out,imag_out,n,Pmax);
}

void microkernel_float_generic::
solid2buf( const float *const *solids, const size_t *P,
                 float *real_out, float *imag_out, size_t n ) const noexcept
{
    solid2buf_impl<float,8>(solids,P,real_out,imag_out,n);
}

void microkernel_float_generic::
buf2solid( const float *real_in, const float *imag_in, 
           float **solids, const size_t *P, size_t n ) const noexcept
{
    buf2solid_impl<float,8>(real_in,imag_in,solids,P,n);
}

void microkernel_double_generic::
euler( const double *x,   const double *y, const double *z,
             double *r,         double *rinv,
             double *cos_alpha, double *sin_alpha,
             double *cos_beta,  double *sin_beta,
             size_t k ) const noexcept
{
    euler_impl<double,4>(x,y,z,r,rinv,cos_alpha,sin_alpha,cos_beta,sin_beta,k);
}

void microkernel_double_generic::
rotscale( const double *cos,      const double *sin,      const double *scale,
          const double *real_in,  const double *imag_in,
                double *real_out,       double *imag_out,
                size_t k, bool forward ) const noexcept
{
    rotscale_impl<double,4>(cos,sin,scale,real_in,imag_in,real_out,imag_out,k,forward);
}

void microkernel_double_generic::
swap( const double *mat, const double *in,
            double *out, size_t k, bool pattern ) const noexcept
{
    swap_impl<double,4,4>(mat,in,out,k,pattern);
}

void microkernel_double_generic::
zm2l( const double *mat, const double *in,
            double *out, size_t k, bool pattern ) const noexcept
{
    zm2l_impl<double,4,4>(mat,in,out,k,pattern);
}

void microkernel_double_generic::
zm2m( const double *mat, const double *in,
            double *out, size_t k ) const noexcept
{
    zm2m_impl<double,4,4>(mat,in,out,k);
}

void microkernel_double_generic::
swap2trans_buf( const double *__restrict__ real_in, 
                const double *__restrict__ imag_in,
                      double **real_out, 
                      double **imag_out,
                      size_t n ) const noexcept
{
    swap2trans_buf_impl<double,4>(real_in,imag_in,real_out,imag_out,n);
}

void microkernel_double_generic::
trans2swap_buf( const double *const *const real_in, 
                const double *const *const imag_in,
                      double *__restrict__ real_out, 
                      double *__restrict__ imag_out,
                      size_t n, size_t Pmax ) const noexcept
{
    trans2swap_buf_impl<double,4>(real_in,imag_in,real_out,imag_out,n,Pmax);
}

void microkernel_double_generic::
solid2buf( const double *const *solids, const size_t *P,
                 double *real_out, double *imag_out, size_t n ) const noexcept
{
    solid2buf_impl<double,4>(solids,P,real_out,imag_out,n);
}

void microkernel_double_generic::
buf2solid( const double *real_in, const double *imag_in, 
           double **solids, const size_t *P, size_t n ) const noexcept
{
    buf2solid_impl<double,4>(real_in,imag_in,solids,P,n);
}

}

