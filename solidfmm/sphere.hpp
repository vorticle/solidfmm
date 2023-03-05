/*
 * Copyright (C) 2023 Matthias Kirchhart
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
#ifndef SOLIDFMM_SPHERE_HPP
#define SOLIDFMM_SPHERE_HPP

#include <cstddef>

namespace solidfmm
{

template <typename real>
struct sphere
{
    real  x { 0 }, y { 0 }, z { 0 }; // Coordinates of the sphere's centre
    real  r { 0 };                   // The sphere's radius
};

extern template struct sphere<float >;
extern template struct sphere<double>;

sphere<float > bounding_sphere( size_t n, const float  *x, const float  *y, const float  *z );
sphere<double> bounding_sphere( size_t n, const double *x, const double *y, const double *z );

sphere<float > bounding_sphere( const sphere<float > *begin, const sphere<float > *end );
sphere<double> bounding_sphere( const sphere<double> *begin, const sphere<double> *end );

}

#endif

