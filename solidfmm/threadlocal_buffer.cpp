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
#include <solidfmm/microkernel.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

#include <new>     // for std::bad_alloc
#include <memory>  // for std::unique_ptr
#include <utility> // for std::swap
#include <cstring> // for std::memcpy
#include <cstdlib> // for std::aligned_alloc

namespace solidfmm
{

template <typename real>
threadlocal_buffer<real>::threadlocal_buffer() noexcept:
P { 0 }, rows { 0 }, cols { 0 }, alignment { 0 }, bufsize { 0 },
buf            { nullptr },
ptr_buf        { nullptr },
r              { nullptr }, rinv           { nullptr },
cos_alpha      { nullptr }, sin_alpha      { nullptr },
cos_beta       { nullptr }, sin_beta       { nullptr },
zeros          { nullptr }, trash          { nullptr },
 swap_real_in  { nullptr },  swap_imag_in  { nullptr },
 swap_real_out { nullptr },  swap_imag_out { nullptr },
trans_real_in  { nullptr }, trans_imag_in  { nullptr },
trans_real_out { nullptr }, trans_imag_out { nullptr },
solid_in       { nullptr }, solid_out      { nullptr },
P_in           { nullptr }, P_out          { nullptr },
x { nullptr }, y { nullptr }, z { nullptr }
{}

template <typename real>
threadlocal_buffer<real>::threadlocal_buffer( const operator_data<real> &op ):
 P         { op.order() }, 
 rows      { op.kernel()->rows }, 
 cols      { op.kernel()->cols },
 alignment { op.kernel()->alignment },
 bufsize   { 0 }
{
    init();
}

template <typename real>
threadlocal_buffer<real>::threadlocal_buffer( size_t p_P, const microkernel<real> *kernel ):
 P         { p_P }, 
 rows      { kernel->rows }, 
 cols      { kernel->cols },
 alignment { kernel->alignment },
 bufsize   { 0 }
{
    init();
}

// This method may only be called by constructors!
template <typename real>
void threadlocal_buffer<real>::init()
{
    std::unique_ptr<real*[]> ptr_buf_ { new real*[4*P + 2*cols] };

    trans_real_in  = ptr_buf_.get() + 0*P;
    trans_real_out = ptr_buf_.get() + 1*P;
    trans_imag_in  = ptr_buf_.get() + 2*P;
    trans_imag_out = ptr_buf_.get() + 3*P;
    solid_in       = ptr_buf_.get() + 4*P;
    solid_out      = ptr_buf_.get() + 4*P + cols;

    // Buffers for x, y, z.
    bufsize  = 3*cols;

    // Size of Euler data buffers.
    bufsize += 6*(P+1)*cols;     

    // Size of swap buffers.
    size_t swap_buf_rows = rows*(1 + (P-1)/rows);
    bufsize += 4*swap_buf_rows*cols; 

    // Size of z-transition buffers.
    for ( size_t m = 0; m < P; ++m )
    {
        // Actual rows in column m of the triangle: P - m.
        size_t padded_rows = (1 + ((P-m-1)/rows)) * rows;
        bufsize += 4*padded_rows*cols;
    }

    // Trash and zeros.
    bufsize += 2*P*(P+1);

    // Does the actual allocation.
    void *mem = std::aligned_alloc( alignment, bufsize * sizeof(real) );
    if ( mem == nullptr ) throw std::bad_alloc {};
    std::memset( mem, 0, bufsize * sizeof(real) );
    std::unique_ptr<real[],decltype(&std::free)> buf_ { reinterpret_cast<real*>(mem), &std::free };
   
    real *tmp = buf_.get(); 
    x             = tmp; tmp += cols;
    y             = tmp; tmp += cols;
    z             = tmp; tmp += cols;
    r             = tmp; tmp += (P+1)*cols;
    rinv          = tmp; tmp += (P+1)*cols; 
    cos_alpha     = tmp; tmp += (P+1)*cols;
    sin_alpha     = tmp; tmp += (P+1)*cols;
    cos_beta      = tmp; tmp += (P+1)*cols;
    sin_beta      = tmp; tmp += (P+1)*cols;
    swap_real_in  = tmp; tmp += swap_buf_rows*cols; 
    swap_imag_in  = tmp; tmp += swap_buf_rows*cols; 
    swap_real_out = tmp; tmp += swap_buf_rows*cols; 
    swap_imag_out = tmp; tmp += swap_buf_rows*cols; 

    for ( size_t m = 0; m < P; ++m )
    {
        // Actual rows in column n of the triangle: P - m.
        // This computes the necessary padding.
        size_t padded_rows = (1 + ((P-m-1)/rows)) * rows;

        trans_real_in [m] = tmp; tmp = tmp + padded_rows*cols; 
        trans_imag_in [m] = tmp; tmp = tmp + padded_rows*cols;
        trans_real_out[m] = tmp; tmp = tmp + padded_rows*cols;
        trans_imag_out[m] = tmp; tmp = tmp + padded_rows*cols;
    } 

    zeros = tmp; tmp = tmp + P*(P+1);
    trash = tmp; tmp = tmp + P*(P+1);

    std::unique_ptr<size_t[]> Pbuf_ { new size_t [ 2*cols ] };

    P_in  = Pbuf_.get();
    P_out = Pbuf_.get() + cols;

        buf =     buf_.release();
    ptr_buf = ptr_buf_.release();
       Pbuf =    Pbuf_.release();
}


template <typename real>
threadlocal_buffer<real>::threadlocal_buffer( const threadlocal_buffer &rhs ):
 P         { rhs.P }, 
 rows      { rhs.rows }, 
 cols      { rhs.cols },
 alignment { rhs.alignment },
 bufsize   { rhs.bufsize   },
 buf       { nullptr       },
 ptr_buf   { nullptr       },
 Pbuf      { nullptr       }
{
    if ( P == 0 )
    {
        P = rows = cols = alignment = bufsize = 0;
        buf            = nullptr;
        ptr_buf        = nullptr;
        Pbuf           = nullptr;
        r              = nullptr; rinv           = nullptr;
        cos_alpha      = nullptr; sin_alpha      = nullptr;
        cos_beta       = nullptr; sin_beta       = nullptr;
        zeros          = nullptr; trash          = nullptr;
         swap_real_in  = nullptr;  swap_imag_in  = nullptr;
         swap_real_out = nullptr;  swap_imag_out = nullptr;
        trans_real_in  = nullptr; trans_imag_in  = nullptr;
        trans_real_out = nullptr; trans_imag_out = nullptr;
        solid_in       = nullptr; solid_out      = nullptr;
        x = y = z      = nullptr; P_in = P_out   = nullptr;
        return;
    }

    std::unique_ptr<real*[]> ptr_buf_ { new real*[4*P + 2*cols] };

    void *mem = std::aligned_alloc( alignment, bufsize * sizeof(real) );
    if ( mem == nullptr ) throw std::bad_alloc {};
    std::memcpy( mem, rhs.buf, bufsize * sizeof(real) );
    std::unique_ptr<real[],decltype(&std::free)> buf_ { reinterpret_cast<real*>(mem), std::free };

    trans_real_in  = ptr_buf_.get() + 0*P;
    trans_real_out = ptr_buf_.get() + 1*P;
    trans_imag_in  = ptr_buf_.get() + 2*P;
    trans_imag_out = ptr_buf_.get() + 3*P;
    solid_in       = ptr_buf_.get() + 4*P;
    solid_out      = ptr_buf_.get() + 4*P + cols;

    x             = buf_.get() + (rhs.x             - rhs.buf);
    y             = buf_.get() + (rhs.y             - rhs.buf);
    z             = buf_.get() + (rhs.z             - rhs.buf);
    r             = buf_.get() + (rhs.r             - rhs.buf);
    rinv          = buf_.get() + (rhs.rinv          - rhs.buf);
    cos_alpha     = buf_.get() + (rhs.cos_alpha     - rhs.buf);
    sin_alpha     = buf_.get() + (rhs.sin_alpha     - rhs.buf);
    cos_beta      = buf_.get() + (rhs.cos_beta      - rhs.buf);
    sin_beta      = buf_.get() + (rhs.sin_beta      - rhs.buf);
    swap_real_in  = buf_.get() + (rhs.swap_real_in  - rhs.buf);
    swap_imag_in  = buf_.get() + (rhs.swap_imag_in  - rhs.buf);
    swap_real_out = buf_.get() + (rhs.swap_real_out - rhs.buf);
    swap_imag_out = buf_.get() + (rhs.swap_imag_out - rhs.buf);

    for ( size_t m = 0; m < P; ++m )
    {
        trans_real_in [m] = buf_.get() + ( rhs.trans_real_in [m] - rhs.buf );
        trans_imag_in [m] = buf_.get() + ( rhs.trans_imag_in [m] - rhs.buf );
        trans_real_out[m] = buf_.get() + ( rhs.trans_real_out[m] - rhs.buf );
        trans_imag_out[m] = buf_.get() + ( rhs.trans_imag_out[m] - rhs.buf );
    }

    zeros = buf_.get() + ( rhs.zeros - rhs.buf );
    trash = buf_.get() + ( rhs.trash - rhs.buf );

    std::unique_ptr<size_t[]> Pbuf_ { new size_t [ 2*cols ] };
    std::memcpy(Pbuf_.get(), rhs.Pbuf, 2*cols*sizeof(size_t));

    P_in  = Pbuf_.get();
    P_out = Pbuf_.get() + cols;

    ptr_buf = ptr_buf_.release();
        buf =     buf_.release();
       Pbuf =    Pbuf_.release();
}

template <typename real>
threadlocal_buffer<real>&
threadlocal_buffer<real>::operator=( const threadlocal_buffer &rhs )
{
    if ( this != &rhs )
    {
        threadlocal_buffer tmp { rhs };
        swap(tmp);
    }
    return *this;
}

template <typename real>
threadlocal_buffer<real>&
threadlocal_buffer<real>::operator=( threadlocal_buffer &&rhs ) noexcept
{
    swap(rhs);
    return *this;
}

template <typename real>
void threadlocal_buffer<real>::swap( threadlocal_buffer &rhs ) noexcept
{
    std::swap(P             ,rhs.P             );
    std::swap(rows          ,rhs.rows          );
    std::swap(cols          ,rhs.cols          );
    std::swap(alignment     ,rhs.alignment     );
    std::swap(bufsize       ,rhs.bufsize       );
    std::swap(buf           ,rhs.buf           );
    std::swap(ptr_buf       ,rhs.ptr_buf       );
    std::swap(Pbuf          ,rhs.Pbuf          );
    std::swap(x             ,rhs.x             );
    std::swap(y             ,rhs.y             );
    std::swap(z             ,rhs.z             );
    std::swap(r             ,rhs.r             );
    std::swap(rinv          ,rhs.rinv          );
    std::swap(cos_alpha     ,rhs.cos_alpha     );
    std::swap(sin_alpha     ,rhs.sin_alpha     );
    std::swap(cos_beta      ,rhs.cos_beta      );
    std::swap(sin_beta      ,rhs.sin_beta      );
    std::swap(trash         ,rhs.trash         );
    std::swap(zeros         ,rhs.zeros         );
    std::swap( swap_real_in ,rhs. swap_real_in );
    std::swap( swap_imag_in ,rhs. swap_imag_in );
    std::swap( swap_real_out,rhs. swap_real_out);
    std::swap( swap_imag_out,rhs. swap_imag_out);
    std::swap(trans_real_in ,rhs.trans_real_in );
    std::swap(trans_imag_in ,rhs.trans_imag_in );
    std::swap(trans_real_out,rhs.trans_real_out);
    std::swap(trans_imag_out,rhs.trans_imag_out);
    std::swap(solid_in      ,rhs.solid_in      );
    std::swap(solid_out     ,rhs.solid_out     );
    std::swap(P_in          ,rhs.P_in          );
    std::swap(P_out         ,rhs.P_out         );
}

template <typename real>
threadlocal_buffer<real>::~threadlocal_buffer()
{
    std::free(buf);
    delete [] ptr_buf;
    delete [] Pbuf;
}

template <typename real>
void threadlocal_buffer<real>::initialise( const operator_data<real> &op )
{
    threadlocal_buffer<real> tmp { op };
    this->swap(tmp);
}

template <typename real>
bool threadlocal_buffer<real>::compatible( const operator_data<real> &op ) const noexcept
{
    if ( P < op.order() ) return false;
    if ( rows != op.kernel()->rows ) return false;
    if ( cols != op.kernel()->cols ) return false;
    if ( alignment != op.kernel()->alignment ) return false;
    return true;
}


// Explicit instatiations.
template class threadlocal_buffer<float>;
template class threadlocal_buffer<double>;

}

