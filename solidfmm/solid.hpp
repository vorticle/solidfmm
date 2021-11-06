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
#ifndef SOLIDFMM_SOLID_HPP
#define SOLIDFMM_SOLID_HPP

#include <cstddef>

namespace solidfmm
{

template <typename real>
class solid
{
public:
    solid() = default;
    solid( size_t P, size_t dim = 1 );
    solid( const solid  &rhs );
    solid(       solid &&rhs ) noexcept;
    solid& operator=( const solid  &rhs );
    solid& operator=(       solid &&rhs ) noexcept;
   ~solid();

    void resize( size_t P, size_t dim = 1 );
    void reinit( size_t P, size_t dim = 1 );
    void zeros() noexcept;

    const real& re( size_t n, size_t m, size_t d = 0 ) const noexcept { return data[ ridx(n,m,d) ]; }
          real& re( size_t n, size_t m, size_t d = 0 )       noexcept { return data[ ridx(n,m,d) ]; }

    const real& im( size_t n, size_t m, size_t d = 0 ) const noexcept { return data[ iidx(n,m,d) ]; }
          real& im( size_t n, size_t m, size_t d = 0 )       noexcept { return data[ iidx(n,m,d) ]; }

    size_t       order    () const noexcept { return P;   }
    size_t       dimension() const noexcept { return dim; }

    const real*  memptr() const noexcept { return data; }
          real*  memptr()       noexcept { return data; }

private:
    size_t dim_stride() const noexcept { return P*(P+1); }
    size_t num_bytes()  const noexcept { return sizeof(real)*dim*P*(P+1); }
    size_t ridx( size_t n, size_t m, size_t d = 0 ) const noexcept { return d*P*(P+1) + (n  )*(n+1) + m; }
    size_t iidx( size_t n, size_t m, size_t d = 0 ) const noexcept { return d*P*(P+1) + (n+1)*(n+1) + m; }

private:
    size_t  P     { 0 };
    size_t  dim   { 1 };
    real   *data  { nullptr };
};

extern template class solid<float>;
extern template class solid<double>;

void dot( const solid<float > &A, const solid<float > &B, float  *result ) noexcept;
void dot( const solid<double> &A, const solid<double> &B, double *result ) noexcept;

void fmadd( const float   fac, const solid<float > &A, solid<float > &B );
void fmadd( const float  *fac, const solid<float > &A, solid<float > &B );
void fmadd( const double  fac, const solid<double> &A, solid<double> &B );
void fmadd( const double *fac, const solid<double> &A, solid<double> &B );

}

#endif

