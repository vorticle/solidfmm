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
#ifndef SOLIDFMM_MICROKERNEL_AVX2_HPP
#define SOLIDFMM_MICROKERNEL_AVX2_HPP

#ifdef __x86_64__
#include <solidfmm/microkernel_avx.hpp>


namespace solidfmm
{

class microkernel_float_avx2: public microkernel_float_avx
{
public:
    static bool available() noexcept;

    microkernel_float_avx2();

    virtual void buf2solid( const float  *real_in, const float *imag_in, 
                                  float **solids, const size_t *P,
                            size_t n ) const noexcept override;
};


class microkernel_double_avx2: public microkernel_double_avx
{
public:
    static bool available() noexcept;

    microkernel_double_avx2();

    virtual void buf2solid( const double  *real_in, const double *imag_in, 
                                  double **solids,  const size_t *P,
                            size_t n ) const noexcept override;
};

}

#endif
#endif

