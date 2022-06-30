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
#include <solidfmm/microkernel_avx.hpp>
#include <solidfmm/microkernel_avx512f.hpp>
#include <solidfmm/microkernel_generic.hpp>

namespace solidfmm
{

// Explicit instantiations.
template class microkernel<float >;
template class microkernel<double>;

template <>
microkernel<float>* get_microkernel<float>()
{

#ifdef __x86_64__
    
         if ( microkernel_float_avx512f::available() ) return new microkernel_float_avx512f;
    else if ( microkernel_float_avx    ::available() ) return new microkernel_float_avx;
    else return new microkernel_float_generic;

#else

    return new microkernel_float_generic {};

#endif

}

template <>
microkernel<double>* get_microkernel<double>()
{

#ifdef __x86_64__

         if ( microkernel_double_avx512f::available() ) return new microkernel_double_avx512f;
    else if ( microkernel_double_avx    ::available() ) return new microkernel_double_avx;
    else return new microkernel_double_generic;

#else

    return new microkernel_double_generic {};

#endif

}

}

