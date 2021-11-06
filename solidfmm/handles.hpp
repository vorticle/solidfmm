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
#ifndef SOLIDFMM_HANDLES_HPP
#define SOLIDFMM_HANDLES_HPP

#include <cstddef> // Needed for size_t.

namespace solidfmm
{

template <typename real> class operator_data;
template <typename real> class threadlocal_buffer;

template <typename real>
class operator_handle
{
public:
    operator_handle();
    operator_handle( size_t P );
   ~operator_handle();
    operator_handle( const operator_handle  &rhs );
    operator_handle& operator=( const operator_handle  &rhs );

    operator_handle( operator_handle &&rhs ) noexcept:
    p { rhs.p }
    {
        rhs.p = nullptr;
    }

    operator_handle& operator=( operator_handle &&rhs ) noexcept
    {
        pointer tmp = p;
        p = rhs.p;
        rhs.p = tmp;
        return *this;
    }

    using       reference =       operator_data<real>&;
    using const_reference = const operator_data<real>&;
    using       pointer   =       operator_data<real>*;
    using const_pointer   = const operator_data<real>*;

    operator       reference()       noexcept { return *p; }
    operator const_reference() const noexcept { return *p; }
    operator       pointer  ()       noexcept { return  p; }
    operator const_pointer  () const noexcept { return  p; }

private:
    pointer p;
};

template <typename real>
class buffer_handle
{
public:
    buffer_handle();
    buffer_handle( const operator_data<real> &op );
    buffer_handle( const buffer_handle  &rhs );
   ~buffer_handle();
    buffer_handle& operator=( const buffer_handle &rhs );

    buffer_handle( buffer_handle &&rhs ) noexcept:
    p { rhs.p }
    {
        rhs.p = nullptr;
    }

    buffer_handle& operator=( buffer_handle &&rhs ) noexcept
    {
        pointer tmp = p;
        p = rhs.p;
        rhs.p = tmp;
        return *this;
    }
    
    using       reference =       threadlocal_buffer<real>&;
    using const_reference = const threadlocal_buffer<real>&;
    using       pointer   =       threadlocal_buffer<real>*;
    using const_pointer   = const threadlocal_buffer<real>*;

    operator       reference()       noexcept { return *p; }
    operator const_reference() const noexcept { return *p; }
    operator       pointer  ()       noexcept { return  p; }
    operator const_pointer  () const noexcept { return  p; }

private:
    pointer p;
};

extern template class operator_handle<float >;
extern template class   buffer_handle<float >;

extern template class operator_handle<double>;
extern template class   buffer_handle<double>;

}

#endif

