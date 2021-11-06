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
#ifndef SOLIDFMM_HARMONICS_HPP
#define SOLIDFMM_HARMONICS_HPP

#include <cstddef>

namespace solidfmm
{

template <typename real> class solid;

namespace harmonics
{

template <typename real> solid<real>  R( size_t P, real x, real y, real z );
template <typename real> solid<real>  S( size_t P, real x, real y, real z );
template <typename real> solid<real> dR( size_t P, real x, real y, real z );
template <typename real> solid<real> dS( size_t P, real x, real y, real z );

template <> solid<float >  R( size_t P, float  x, float  y, float  z );
template <> solid<float >  S( size_t P, float  x, float  y, float  z );
template <> solid<float > dR( size_t P, float  x, float  y, float  z );
template <> solid<float > dS( size_t P, float  x, float  y, float  z );

template <> solid<double>  R( size_t P, double x, double y, double z );
template <> solid<double>  S( size_t P, double x, double y, double z );
template <> solid<double> dR( size_t P, double x, double y, double z );
template <> solid<double> dS( size_t P, double x, double y, double z );

}

}

#endif

