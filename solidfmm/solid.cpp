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
#include <solidfmm/solid.hpp>

#include <utility>   // swap
#include <cstdlib>   // malloc,realloc,free
#include <cstring>   // memcpy,memset
#include <stdexcept> // bad_alloc, logic_error

#pragma GCC optimize("O2")
#pragma GCC optimize("tree-vectorize")

namespace solidfmm
{

namespace
{

template <typename real>
void dot_impl( const solid<real> &A, const solid<real> &B, real *result ) noexcept
{
    using std::min;
    const size_t PA       { A.order()  };
    const size_t PB       { B.order()  };
    const size_t P        { min(PA,PB) };
    const size_t N        { P *(P +1)  };
    const size_t Astride  { PA*(PA+1)  };
    const size_t Bstride  { PB*(PB+1)  };

    for ( size_t da = 0; da < A.dimension(); ++da )
    for ( size_t db = 0; db < B.dimension(); ++db )
    {
        real tmp { 0 };
        const real *a { A.memptr() + Astride*da };
        const real *b { B.memptr() + Bstride*db };
        for ( size_t k = 0; k < N; ++k )
            tmp += a[k]*b[k];
       
        real tmp2 { 0 };
        for ( size_t k = 0; k < P; ++k )
            tmp2 += A.re(k,0)*B.re(k,0);

        *result++ = real(2)*tmp - tmp2;
    }
}

template <typename real>
void fmadd_impl( const real fac, const solid<real> &A, solid<real> &B )
{
    if ( A.dimension() != B.dimension() )
        throw std::logic_error { "solidfmm::fmadd(): dimension mismatch." };

    if ( A.order() != B.order() )
        throw std::logic_error { "solidfmm::fmadd(): order mismatch." };

    size_t P = A.order();
    size_t D = A.dimension();
    const real *in  = A.memptr();
          real *out = B.memptr();
    for ( size_t i = 0; i < D*P*(P+1); ++i )
        out[ i ] += fac*in[ i ];
}

template <typename real>
void fmadd_impl( const real *fac, const solid<real> &A, solid<real> &B )
{
    if ( A.dimension() != B.dimension() )
        throw std::logic_error { "solidfmm::fmadd(): dimension mismatch." };

    if ( A.order() != B.order() )
        throw std::logic_error { "solidfmm::fmadd(): order mismatch." };

    size_t P = A.order();
    size_t D = A.dimension();
    for ( size_t d = 0; d < D; ++d )
    {
        const real *in  = A.memptr() + d*P*(P+1);
              real *out = B.memptr() + d*P*(P+1);

        for ( size_t i = 0; i < P*(P+1); ++i )
            out[ i ] += fac[ d ]*in[ i ];
    }
}
  
}

template <typename real>
solid<real>::solid( size_t arg_P, size_t arg_dim ):
   P { 0 },
 dim { 1 },
data { nullptr } 
{
    resize(arg_P,arg_dim);
    zeros();
}

template <typename real>
solid<real>::solid( const solid &rhs ):
   P { 0 },
 dim { 1 },
data { nullptr } 
{
    resize(rhs.P,rhs.dim);
    std::memcpy(data,rhs.data,num_bytes());
}

template <typename real>
solid<real>::solid( solid &&rhs ) noexcept:
   P { rhs.P    },
 dim { rhs.dim  },
data { rhs.data }
{
    rhs.P    = 0;
    rhs.dim  = 1;
    rhs.data = nullptr;
}

template <typename real>
solid<real>& solid<real>::operator=( solid &&rhs ) noexcept
{
   std::swap( P, rhs.P );
   std::swap( dim, rhs.dim );
   std::swap( data, rhs.data );
   return *this;
}

template <typename real>
solid<real>& solid<real>::operator=( const solid &rhs )
{
    if ( this != &rhs )
    {
        resize(rhs.P,rhs.dim);
        std::memcpy(data,rhs.data,num_bytes());
    }
    return *this;
}

template <typename real>
solid<real>::~solid()
{
    std::free(data);
}
 
template <typename real>
void solid<real>::resize( size_t arg_P, size_t arg_dim )
{
    size_t new_bytes = sizeof(real)*arg_P*(arg_P+1)*arg_dim;
    size_t old_bytes = num_bytes();
    if ( new_bytes != old_bytes )
    {
        if ( new_bytes )
        {
            void *tmp = std::realloc( data, new_bytes );
            if ( tmp == nullptr ) throw std::bad_alloc {}; 
            else data = reinterpret_cast<real*>(tmp);
        }
        else
        {
            std::free(data);
            data = nullptr;
        }
    }

    P    = arg_P;
    dim  = arg_dim;
}

template <typename real>
void solid<real>::reinit( size_t arg_P, size_t arg_dim )
{
    resize(arg_P,arg_dim);
    zeros();
}

template <typename real>
void solid<real>::zeros() noexcept
{
    std::memset(data,0,num_bytes());
}

void dot( const solid<float> &A, const solid<float> &B, float *result ) noexcept
{
    dot_impl<float>(A,B,result);
}

void dot( const solid<double> &A, const solid<double> &B, double *result ) noexcept
{
    dot_impl<double>(A,B,result);
}    

void fmadd( const float   fac, const solid<float > &A, solid<float > &B )
{
    fmadd_impl<float>(fac,A,B);
}

void fmadd( const float  *fac, const solid<float > &A, solid<float > &B )
{
    fmadd_impl<float>(fac,A,B);
}

void fmadd( const double  fac, const solid<double> &A, solid<double> &B )
{
    fmadd_impl<double>(fac,A,B);
}

void fmadd( const double *fac, const solid<double> &A, solid<double> &B )
{
    fmadd_impl<double>(fac,A,B);
}

// Explicit instantiations.
template class solid<float>;
template class solid<double>;

}

