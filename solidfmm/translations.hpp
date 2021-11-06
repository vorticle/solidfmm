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
#ifndef SOLIDFMM_TRANSLATIONS_HPP
#define SOLIDFMM_TRANSLATIONS_HPP

#include <cstddef>

namespace solidfmm
{

template <typename real> class solid;
template <typename real> class operator_data;
template <typename real> class threadlocal_buffer;


void m2l( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const M, solid<float> *const *const L,
          const float *x, const float *y, const float *z );

void m2l( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const M, solid<double> *const *const L,
          const double *x, const double *y, const double *z );

void m2m( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const Min, solid<float> *const *const Mout,
          const float *x, const float *y, const float *z );

void m2m( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const Min, solid<double> *const *const Mout,
          const double *x, const double *y, const double *z );

void l2l( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const Min, solid<float> *const *const Mout,
          const float *x, const float *y, const float *z );

void l2l( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const Lin, solid<double> *const *const Lout,
          const double *x, const double *y, const double *z );
}

#endif

