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
#include <solidfmm/swap_matrix.hpp>

#include <new>
#include <utility>
#include <cstring>
#include <cstdlib>

namespace solidfmm
{

template <typename real>
swap_matrix<real>::swap_matrix( int n ):
n_end { 0 }, data { nullptr }
{
    reserve(n);
}

template <typename real>
swap_matrix<real>::swap_matrix( const swap_matrix &rhs ):
n_end { rhs.n_end }, data { nullptr }
{
    if ( n_end > 0 )
    {
        int n    = n_end;
        int size = sizeof(real) * ((n*(4*n*n-1))/3);
        void*  new_data = std::realloc( data, size );
        if ( new_data == nullptr ) throw std::bad_alloc {};
        else data = reinterpret_cast<real*>(new_data);
        std::memcpy( new_data, rhs.data, size );
    }
}

template <typename real>
swap_matrix<real>::swap_matrix( swap_matrix &&rhs ) noexcept:
n_end { rhs.n_end }, data { rhs.data }
{
    rhs.n_end = 0;
    rhs.data  = nullptr;
}

template <typename real>
swap_matrix<real>& swap_matrix<real>::operator=( const swap_matrix &rhs )
{
    if ( &rhs != this )
    {
        if ( rhs.n_end == 0 )
        {
            free(data);
            data  = nullptr;
            n_end = 0;
        }
        else
        {
            int n        = rhs.n_end;
            int size     = sizeof(real) * ((n*(4*n*n-1))/3);
            void*  new_data = std::realloc( data, size );
            if ( new_data == nullptr ) throw std::bad_alloc {};
            else data = reinterpret_cast<real*>(new_data);

            if ( n_end < rhs.n_end )
            {
                int begin = idx(n_end,-n_end,-n_end);
                int end   = idx(rhs.n_end,-rhs.n_end,-rhs.n_end);
                std::memcpy( data + begin, rhs.data + begin, sizeof(real)*(end-begin) );
            }

            n_end = rhs.n_end;
        }
    }

    return *this;
}

template <typename real>
swap_matrix<real>& swap_matrix<real>::operator=( swap_matrix &&rhs ) noexcept
{
    std::swap( n_end, rhs.n_end );
    std::swap( data,  rhs.data  );
    return *this;
}

template <typename real>
swap_matrix<real>::~swap_matrix()
{
    std::free(data);
}

template <typename real> inline
real swap_matrix<real>::operator()( int n, int m, int l ) const noexcept
{
    return data[ idx(n,m,l) ];
}

template <typename real> inline
real swap_matrix<real>::b( int n, int m, int l ) const noexcept
{
    if ( abs(l) > n ) return 0;
    else return data[ idx(n,m,l) ];
}

template <typename real>
void swap_matrix<real>::reserve( int n )
{
    if ( n > n_end )
    {
        void* new_data = std::realloc( data, sizeof(real) * ((n*(4*n*n-1))/3) );
        if ( new_data == nullptr ) throw std::bad_alloc {};
        else data = reinterpret_cast<real*>(new_data);

        while ( n > n_end )
            compute_matrix(n_end++);
    }
}

template <typename real> inline
int swap_matrix<real>::order() const noexcept
{
    return n_end;
}

template <typename real> constexpr
int swap_matrix<real>::idx( int n, int m, int l ) const noexcept
{
    return (n*(4*n*n-1))/3 + (m+n) + (l+n)*(2*n+1);
}

// Assuming all matrices up to level n - 1 have been computed,
// this routine computes the matrix for the given level n.
template <typename real>
void swap_matrix<real>::compute_matrix( int n ) noexcept
{
    if ( n == 0 )
    {
        data[0] = 1;
    }
    else
    {
        // First recursion for m = 0.
        for ( int l = -n; l <= n; ++l )
        {
            data[ idx(n,0,l) ] = 0.5*( b(n-1,0,l-1) - b(n-1,0,l+1) );
        }

        // Second recursion for all other m.
        for ( int m = 0;  m <  n; ++m )
        for ( int l = -n; l <= n; ++l )
        {
            data[ idx(n, m+1,l) ] = 0.5*( b(n-1, m,l-1) + b(n-1, m,l+1) ) + b(n-1, m,l);
            data[ idx(n,-m-1,l) ] = 0.5*( b(n-1,-m,l-1) + b(n-1,-m,l+1) ) - b(n-1,-m,l);
        }
    }
}

template class swap_matrix<float>;
template class swap_matrix<double>;

}

