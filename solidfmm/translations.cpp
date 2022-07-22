/*
 * Copyright (C) 2021, 2022 Matthias Kirchhart
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
#include <solidfmm/translations.hpp>

#include <solidfmm/solid.hpp>
#include <solidfmm/microkernel.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

#include <algorithm> // For std::min
#include <stdexcept>

namespace solidfmm
{

namespace
{

template <typename real>
void forward_transform_M( const operator_data<real> &op, threadlocal_buffer<real> &buf, size_t n ) noexcept
{
    // scale-rot(alpha)-swap-rot(beta)-swap
    const microkernel<real> *const kernel  { op.kernel() };
    const size_t cols    { kernel->cols };
    const size_t rows    { kernel->rows };
    const bool   pattern { !(n&1) };

    kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.rinv + (n+1)*cols,
                      buf.swap_real_in , buf.swap_imag_in ,
                      buf.swap_real_out, buf.swap_imag_out,
                      n+1, true );

    std::swap( buf.swap_real_out, buf.swap_real_in );
    std::swap( buf.swap_imag_out, buf.swap_imag_in );

    const real *const real_mat { op.real_swap_matrix(n) };
    const real *const imag_mat { op.imag_swap_matrix(n) };
    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1,  pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_beta, buf.sin_beta, buf.rinv,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in,  buf.swap_imag_in,
                      n+1, true );
  
    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }
}

template <typename real>
void backward_transform_M( const operator_data<real> &op, threadlocal_buffer<real> &buf, size_t n ) noexcept
{
    // swap-rot(-beta)-swap-rot(-alpha)-rescale 
    const microkernel<real> *const kernel  { op.kernel() };
    const size_t cols    { kernel->cols };
    const size_t rows    { kernel->rows };
    const bool   pattern { !(n&1) };

    const real *const real_mat { op.real_swap_matrix(n) };
    const real *const imag_mat { op.imag_swap_matrix(n) };
    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_beta, buf.sin_beta, buf.rinv,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in , buf.swap_imag_in ,
                      n+1, false );

    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1,  pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.r + (n+1)*cols,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in,  buf.swap_imag_in,
                      n+1, false );
  
    std::swap( buf.swap_real_out, buf.swap_real_in );
    std::swap( buf.swap_imag_out, buf.swap_imag_in );
}

template <typename real>
void backward_transform_L( const operator_data<real> &op, threadlocal_buffer<real> &buf, size_t n ) noexcept
{
    // swap-rot(-beta)-swap-rot(-alpha)-invscale
    const microkernel<real> *const kernel  { op.kernel() };
    const size_t cols { kernel->cols };
    const size_t rows { kernel->rows };
    const bool   pattern { !(n&1) };

    const real *const real_mat { op.real_swap_matrix_transposed(n) };
    const real *const imag_mat { op.imag_swap_matrix_transposed(n) };
    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_beta, buf.sin_beta, buf.rinv,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in,  buf.swap_imag_in,
                      n+1, false );

    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.rinv + n*cols,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in,  buf.swap_imag_in ,
                      n+1, false );

    std::swap( buf.swap_real_out, buf.swap_real_in );
    std::swap( buf.swap_imag_out, buf.swap_imag_in );
}

template <typename real>
void forward_transform_L( const operator_data<real> &op, threadlocal_buffer<real> &buf, size_t n ) noexcept
{
    // scale-rot(alpha)-swap-rot(beta) 
    const microkernel<real> *const kernel  { op.kernel() };
    const size_t cols { kernel->cols };
    const size_t rows { kernel->rows };
    const bool   pattern { !(n&1) };

    kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.r + n*cols,
                      buf.swap_real_in , buf.swap_imag_in ,
                      buf.swap_real_out, buf.swap_imag_out,
                      n+1, true );

    std::swap( buf.swap_real_out, buf.swap_real_in );
    std::swap( buf.swap_imag_out, buf.swap_imag_in );

    const real *const real_mat { op.real_swap_matrix_transposed(n) };
    const real *const imag_mat { op.imag_swap_matrix_transposed(n) };
    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

    kernel->rotscale( buf.cos_beta, buf.sin_beta, buf.r,
                      buf.swap_real_out, buf.swap_imag_out,
                      buf.swap_real_in,  buf.swap_imag_in,
                      n+1, true );

    for ( size_t m = 0; m <= n; m += rows )
    {
        kernel->swap( real_mat + (m/2)*(n+1), 
                      buf.swap_real_in, 
                      buf.swap_real_out + m*cols,
                      n + 1, pattern );

        kernel->swap( imag_mat + (m/2)*(n+1), 
                      buf.swap_imag_in,
                      buf.swap_imag_out + m*cols, 
                      n + 1, !pattern );
    }

}


template <typename real>
void m2l_impl( const operator_data<real> &op, threadlocal_buffer<real> &buf )
{
    const microkernel<real> *const kernel { op.kernel() };

    const size_t cols { kernel->cols };
    const size_t rows { kernel->rows };
    size_t Pmax { 0 }, Pmax_in { 0 }, Pmax_out { 0 };
    for ( size_t l = 0; l < cols; ++l ) 
    {
        Pmax_in  = std::max( Pmax_in,  buf.P_in [l] );
        Pmax_out = std::max( Pmax_out, buf.P_out[l] );
    }
    Pmax = std::max( Pmax_in, Pmax_out );

    kernel->euler( buf.x, buf.y, buf.z,
                   buf.r, buf.rinv,
                   buf.cos_alpha, buf.sin_alpha,
                   buf.cos_beta , buf.sin_beta , Pmax + 1 );

    ///////////////////////////
    // Coordinate transform. //
    ///////////////////////////
    
    for ( size_t n = 0; n <  Pmax_in; ++n )
    {
        // Copy data from solids into buffer.
        kernel->solid2buf( buf.solid_in, buf.zeros, buf.P_in, buf.swap_real_in, buf.swap_imag_in, n );
        
        // Transformation 
        forward_transform_M( op, buf, n );
    
        // Copy into z-translation buffer.
        kernel->swap2trans_buf( buf. swap_real_out, buf. swap_imag_out,
                                buf.trans_real_in , buf.trans_imag_in, n );
    }

    ////////////////////
    // z-translation. //
    ////////////////////

    for ( size_t m = 0; m < std::min(Pmax_in,Pmax_out); ++m )
    for ( size_t n = m; n < Pmax_out; n += rows )
    {
        bool pattern = ((n+m) & 1);
        kernel->zm2l( op.faculties() + (n+m),
                      buf.trans_real_in [m],
                      buf.trans_real_out[m] + (n-m)*cols,
                      Pmax_in - m, pattern );

        kernel->zm2l( op.faculties() + (n+m), 
                      buf.trans_imag_in [m],
                      buf.trans_imag_out[m] + (n-m)*cols, 
                      Pmax_in - m, pattern );
    }

    ///////////////////////////////////
    // Inverse coordinate transform. //
    ///////////////////////////////////

    for ( size_t n = 0; n < Pmax_out; ++n )
    {
        // Copy into transformation buffer
        kernel->trans2swap_buf( buf.trans_real_out, buf.trans_imag_out,
                                buf. swap_real_in,  buf. swap_imag_in,
                                n, Pmax_in );

        // Transformation
        backward_transform_L( op, buf, n );

        // Add result to output solids.
        kernel->buf2solid( buf.swap_real_out, buf.swap_imag_out,
                           buf.solid_out, buf.trash, buf.P_out, n );
    }
}

template <typename real>
void m2m_impl( const operator_data<real> &op, threadlocal_buffer<real> &buf )
{
    const microkernel<real> *const kernel { op.kernel() };

    const size_t cols { kernel->cols };
    const size_t rows { kernel->rows };
    size_t Pmax { 0 }, Pmax_in { 0 }, Pmax_out { 0 };
    for ( size_t l = 0; l < cols; ++l ) 
    {
        Pmax_in  = std::max( Pmax_in,  buf.P_in [l] );
        Pmax_out = std::max( Pmax_out, buf.P_out[l] );
 
        buf.x[l] = -buf.x[l];
        buf.y[l] = -buf.y[l];
        buf.z[l] = -buf.z[l];
    }
    Pmax = std::max( Pmax_in, Pmax_out );

    kernel->euler( buf.x, buf.y, buf.z,
                   buf.r, buf.rinv,
                   buf.cos_alpha, buf.sin_alpha,
                   buf.cos_beta , buf.sin_beta , Pmax + 1 );

    ///////////////////////////
    // Coordinate transform. //
    ///////////////////////////
    
    for ( size_t n = 0; n <  Pmax_in; ++n )
    {
        // Copy data from solids into buffer.
        kernel->solid2buf( buf.solid_in, buf.zeros, buf.P_in, buf.swap_real_in, buf.swap_imag_in, n );
        
        // Transformation 
        forward_transform_M( op, buf, n );
    
        // Copy into z-translation buffer.
        kernel->swap2trans_buf( buf. swap_real_out, buf. swap_imag_out,
                                buf.trans_real_in , buf.trans_imag_in, n );
    }

    ////////////////////
    // z-translation. //
    ////////////////////

    for ( size_t m = 0; m < std::min(Pmax_in,Pmax_out); ++m )
    for ( size_t n = m; n < Pmax_out; n += rows )
    {
        size_t lower_bound = m;
        size_t upper_bound = std::min(n+rows,Pmax_in);
        size_t k = upper_bound - lower_bound;

        kernel->zm2m( op.inverse_faculties_decreasing(n+rows-1-m),
                      buf.trans_real_in [m],
                      buf.trans_real_out[m] + (n-m)*cols, k );

        kernel->zm2m( op.inverse_faculties_decreasing(n+rows-1-m), 
                      buf.trans_imag_in [m],
                      buf.trans_imag_out[m] + (n-m)*cols, k );
    }

    ///////////////////////////////////
    // Inverse coordinate transform. //
    ///////////////////////////////////

    for ( size_t n = 0; n < Pmax_out; ++n )
    {
        // Copy into transformation buffer
        kernel->trans2swap_buf( buf.trans_real_out, buf.trans_imag_out,
                                buf. swap_real_in,  buf. swap_imag_in,
                                n, Pmax_in );

        // Transformation
        backward_transform_M( op, buf, n );

        // Add result to output solids.
        kernel->buf2solid( buf.swap_real_out, buf.swap_imag_out,
                           buf.solid_out, buf.trash, buf.P_out, n );
    }
}

template <typename real>
void l2l_impl( const operator_data<real> &op, threadlocal_buffer<real> &buf )
{
    const microkernel<real> *const kernel { op.kernel() };

    const size_t cols { kernel->cols };
    const size_t rows { kernel->rows };
    size_t Pend { 0 }, Pmax_in { 0 }, Pmax_out { 0 };
    for ( size_t l = 0; l < cols; ++l ) 
    {
        Pmax_in  = std::max( Pmax_in,  buf.P_in [l] );
        Pmax_out = std::max( Pmax_out, buf.P_out[l] );
    }
    Pend = std::min(Pmax_in,Pmax_out);

    kernel->euler( buf.x, buf.y, buf.z,
                   buf.r, buf.rinv,
                   buf.cos_alpha, buf.sin_alpha,
                   buf.cos_beta , buf.sin_beta , Pend + 1 );

    ///////////////////////////
    // Coordinate transform. //
    ///////////////////////////
    
    for ( size_t n = 0; n < Pmax_in; ++n )
    {
        // Copy data from solids into buffer.
        kernel->solid2buf( buf.solid_in, buf.zeros, buf.P_in, buf.swap_real_in, buf.swap_imag_in, n );
        
        // Transformation 
        forward_transform_L( op, buf, n );
    
        // Copy into z-translation buffer.
        kernel->swap2trans_buf( buf. swap_real_out, buf. swap_imag_out,
                                buf.trans_real_in , buf.trans_imag_in, n );
    }

    ////////////////////
    // z-translation. //
    ////////////////////

    for ( size_t m = 0; m < Pend; ++m )
    for ( size_t n = m; n < Pend; n += rows )
    {
        const real *fac = op.inverse_faculties_increasing(0) - (rows-1);
        kernel->zm2m( fac,
                      buf.trans_real_in [m] + (n-m)*cols,
                      buf.trans_real_out[m] + (n-m)*cols, Pmax_in - n );

        kernel->zm2m( fac,
                      buf.trans_imag_in [m] + (n-m)*cols,
                      buf.trans_imag_out[m] + (n-m)*cols, Pmax_in - n );
    }

    ///////////////////////////////////
    // Inverse coordinate transform. //
    ///////////////////////////////////

    for ( size_t n = 0; n < Pend; ++n )
    {
        // Copy into transformation buffer
        kernel->trans2swap_buf( buf.trans_real_out, buf.trans_imag_out,
                                buf. swap_real_in,  buf. swap_imag_in,
                                n, Pend );

        // Transformation
        backward_transform_L( op, buf, n );

        // Add result to output solids.
        kernel->buf2solid( buf.swap_real_out, buf.swap_imag_out,
                           buf.solid_out, buf.trash, buf.P_out, n );
    }
}

template <typename real>
class m2l_queue
{
private:
    const      operator_data<real> &op;
          threadlocal_buffer<real> &buf;
    size_t size, cols;

public:
    m2l_queue() = delete;
    m2l_queue( const operator_data<real> &arg_op, threadlocal_buffer<real> &arg_buf ):
    op { arg_op }, buf { arg_buf }, size { 0 }, cols { op.kernel()->cols }
    {}
    
    ~m2l_queue() { flush(); }

    void push( const real *M, size_t PM,
                     real *L, size_t PL,
               real x, real y, real z ) noexcept
    {
        if ( size == cols )
            flush();

        buf.x        [ size ] = x;
        buf.y        [ size ] = y;
        buf.z        [ size ] = z;
        buf.P_in     [ size ] = PM;
        buf.P_out    [ size ] = PL;
        buf.solid_in [ size ] = const_cast<real*>(M);
        buf.solid_out[ size ] = L;
        size++;
    }

    void flush() noexcept
    {
        if ( size )
        {
            for ( size_t i = size; i < cols; ++i )
                buf.P_in[i] = buf.P_out[i] = 0;

            m2l_impl<real>(op,buf);
            size = 0;
        }
    }
};

template <typename real>
class m2m_queue
{
private:
    const      operator_data<real> &op;
          threadlocal_buffer<real> &buf;
    size_t size, cols;

public:
    m2m_queue() = delete;
    m2m_queue( const operator_data<real> &arg_op, threadlocal_buffer<real> &arg_buf ):
    op { arg_op }, buf { arg_buf }, size { 0 }, cols { op.kernel()->cols }
    {}
    
    ~m2m_queue() { flush(); }

    void push( const real *Min,  size_t Pin,
                     real *Mout, size_t Pout,
               real x, real y, real z ) noexcept
    {
        real r2 = x*x + y*y + z*z;
        if ( r2 == 0 )
        {
            size_t P = std::min(Pout,Pin);
            for ( size_t i = 0; i < P*(P+1); ++i )
                Mout[i] += Min[i];

            return;
        }

        if ( size == cols )
            flush();

        buf.x        [ size ] = x;
        buf.y        [ size ] = y;
        buf.z        [ size ] = z;
        buf.P_in     [ size ] = Pin;
        buf.P_out    [ size ] = Pout;
        buf.solid_in [ size ] = const_cast<real*>(Min);
        buf.solid_out[ size ] = Mout;
        size++;
    }

    void flush() noexcept
    {
        if ( size )
        {
            for ( size_t i = size; i < cols; ++i )
                buf.P_in[i] = buf.P_out[i] = 0;

            m2m_impl<real>(op,buf);
            size = 0;
        }
    }
};

template <typename real>
class l2l_queue
{
private:
    const      operator_data<real> &op;
          threadlocal_buffer<real> &buf;
    size_t size, cols;

public:
    l2l_queue() = delete;
    l2l_queue( const operator_data<real> &arg_op, threadlocal_buffer<real> &arg_buf ):
    op { arg_op }, buf { arg_buf }, size { 0 }, cols { op.kernel()->cols }
    {}
    
    ~l2l_queue() { flush(); }

    void push( const real *Lin,  size_t Pin,
                     real *Lout, size_t Pout,
               real x, real y, real z ) noexcept
    {

        real r2 = x*x + y*y + z*z;
        if ( r2 == 0 )
        {
            size_t P = std::min(Pout,Pin);
            for ( size_t i = 0; i < P*(P+1); ++i )
                Lout[i] += Lin[i];

            return;
        }

        if ( size == cols )
            flush();

        buf.x        [ size ] = x;
        buf.y        [ size ] = y;
        buf.z        [ size ] = z;
        buf.P_in     [ size ] = Pin;
        buf.P_out    [ size ] = Pout;
        buf.solid_in [ size ] = const_cast<real*>(Lin);
        buf.solid_out[ size ] = Lout;
        size++;
    }

    void flush() noexcept
    {
        if ( size )
        {
            for ( size_t i = size; i < cols; ++i )
                buf.P_in[i] = buf.P_out[i] = 0;

            l2l_impl<real>(op,buf);
            size = 0;
        }
    }
};

}

void m2l( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          size_t howmany, const solid<float> *const *const M, solid<float> *const *const L,
          const float *x, const float *y, const float *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::m2l<float>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( M[i]->dimension() != L[i]->dimension() )
            throw std::logic_error { "solidfmm::m2l<float>(): dimension mismatch." };

        if ( M[i]->order() > op.order() || L[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::m2l<float>(): orders exceeding operator data." };
    }

    m2l_queue<float> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {

        for ( size_t d = 0; d < M[i]->dimension(); ++d )
        {
            queue.push( &(M[i]->re(0,0,d)), M[i]->order(),
                        &(L[i]->re(0,0,d)), L[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}

void m2l( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          size_t howmany, const solid<double> *const *const M, solid<double> *const *const L,
          const double *x, const double *y, const double *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::m2l<double>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( M[i]->dimension() != L[i]->dimension() )
            throw std::logic_error { "solidfmm::m2l<double>(): dimension mismatch." };

        if ( M[i]->order() > op.order() || L[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::m2l<double>(): orders exceeding operator data." };
    }

    m2l_queue<double> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {

        for ( size_t d = 0; d < M[i]->dimension(); ++d )
        {
            queue.push( &(M[i]->re(0,0,d)), M[i]->order(),
                        &(L[i]->re(0,0,d)), L[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}


void m2m( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          size_t howmany, const solid<float> *const *const Min, solid<float> *const *const Mout,
          const float *x, const float *y, const float *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::m2m<float>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( Min[i]->dimension() != Mout[i]->dimension() )
            throw std::logic_error { "solidfmm::m2m<float>(): dimension mismatch." };

        if ( Min[i]->order() > op.order() || Mout[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::m2m<float>(): orders exceeding operator data." };
    }


    m2m_queue<float> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {
        for ( size_t d = 0; d < Min[i]->dimension(); ++d )
        {
            queue.push( &(Min [i]->re(0,0,d)), Min [i]->order(),
                        &(Mout[i]->re(0,0,d)), Mout[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}

void m2m( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          size_t howmany, const solid<double> *const *const Min, solid<double> *const *const Mout,
          const double *x, const double *y, const double *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::m2m<double>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( Min[i]->dimension() != Mout[i]->dimension() )
            throw std::logic_error { "solidfmm::m2m<double>(): dimension mismatch." };

        if ( Min[i]->order() > op.order() || Mout[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::m2m<double>(): orders exceeding operator data." };
    }


    m2m_queue<double> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {
        for ( size_t d = 0; d < Min[i]->dimension(); ++d )
        {
            queue.push( &(Min [i]->re(0,0,d)), Min [i]->order(),
                        &(Mout[i]->re(0,0,d)), Mout[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}

void l2l( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          size_t howmany, const solid<float> *const *const Lin, solid<float> *const *const Lout,
          const float *x, const float *y, const float *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::l2l<float>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( Lin[i]->dimension() != Lout[i]->dimension() )
            throw std::logic_error { "solidfmm::l2l<float>(): dimension mismatch." };

        if ( Lin[i]->order() > op.order() || Lout[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::l2l<float>(): orders exceeding operator data." };
    }

    l2l_queue<float> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {
        for ( size_t d = 0; d < Lin[i]->dimension(); ++d )
        {
            queue.push( &(Lin [i]->re(0,0,d)), Lin [i]->order(),
                        &(Lout[i]->re(0,0,d)), Lout[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}

void l2l( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          size_t howmany, const solid<double> *const *const Lin, solid<double> *const *const Lout,
          const double *x, const double *y, const double *z )
{
    if ( buf.compatible(op) == false )
        throw std::logic_error { "solidfmm::l2l<double>(): operator data and buffer incompatible" };

    for ( size_t i = 0; i < howmany; ++i )
    {
        if ( Lin[i]->dimension() != Lout[i]->dimension() )
            throw std::logic_error { "solidfmm::l2l<double>(): dimension mismatch." };

        if ( Lin[i]->order() > op.order() || Lout[i]->order() > op.order() )
            throw std::out_of_range { "solidfmm::l2l<double>(): orders exceeding operator data." };
    }

    l2l_queue<double> queue(op,buf);
    for ( size_t i = 0; i < howmany; ++i )
    {
        for ( size_t d = 0; d < Lin[i]->dimension(); ++d )
        {
            queue.push( &(Lin [i]->re(0,0,d)), Lin [i]->order(),
                        &(Lout[i]->re(0,0,d)), Lout[i]->order(),
                        x[i], y[i], z[i] );
        }
    }
}

}

