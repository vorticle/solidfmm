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
#include <solidfmm/handles.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

namespace solidfmm
{

namespace
{

template <typename real> constexpr size_t Pdefault();

template <> constexpr size_t Pdefault<float >() { return 18; }
template <> constexpr size_t Pdefault<double>() { return 30; }

}

template <typename real>
operator_handle<real>::operator_handle():
p { new operator_data<real>(Pdefault<real>()) }
{}

template <typename real>
operator_handle<real>::operator_handle( size_t P ):
p { new operator_data<real>(P) }
{}

template <typename real>
operator_handle<real>::~operator_handle()
{
    delete p;
}

template <typename real>
operator_handle<real>::operator_handle( const operator_handle &rhs ):
p { new operator_data<real> { rhs.p->order() } }
{}

template <typename real>
operator_handle<real>&
operator_handle<real>::operator=( const operator_handle &rhs )
{
    if ( &rhs != this )
    {
        pointer tmp = new operator_data<real> { rhs.p->order() };
        delete p;
        p = tmp;
    }
    return *this;
}


template <typename real>
buffer_handle<real>::buffer_handle():
p { new threadlocal_buffer<real> {} }
{}

template <typename real>
buffer_handle<real>::buffer_handle( const operator_data<real> &op ):
p { new threadlocal_buffer<real> { op } }
{}

template <typename real>
buffer_handle<real>::buffer_handle( const buffer_handle &rhs ):
p { new threadlocal_buffer<real> { *rhs.p } }
{}

template <typename real>
buffer_handle<real>::~buffer_handle()
{
    delete p;
}

template <typename real>
buffer_handle<real>& buffer_handle<real>::operator=( const buffer_handle<real> &rhs )
{
    *p = *rhs.p;
    return *this;
}

//////////////////////////////
// Explicit instantiations. //
//////////////////////////////

template class buffer_handle<float >;
template class buffer_handle<double>;
template class operator_handle<float >;
template class operator_handle<double>;

}

