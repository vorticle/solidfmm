/*
 * Copyright (C) 2021 Matthias Kirchhart
 *
 * This file is part of solidfmm.
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
#include <locale>
#include <cstdlib>
#include <iomanip>
#include <complex>
#include <iostream>

#include <solidfmm/solid.hpp>
#include <solidfmm/handles.hpp>
#include <solidfmm/harmonics.hpp>
#include <solidfmm/stopwatch.hpp>
#include <solidfmm/translations.hpp>

using namespace solidfmm;


template <typename real>
real benchmark( size_t P );

int main()
{
    // Somehow it is needed to execute this once before the loop
    // to get clean data for the P = 1 case in the benchmark below.
    benchmark<float>(1);

    std::locale loc {""};
    std::cout.imbue(loc);

    std::cout << std::setw(4)  << "P";
    std::cout << std::setw(25) << "float";
    std::cout << std::setw(25) << "double";
    std::cout << std::endl;

    for ( size_t P = 1; P <= 18; ++P )
    {
        std::cout << std::setw(4)  << P;
        std::cout << std::setw(25) << benchmark<float >(P);
        std::cout << std::setw(25) << benchmark<double>(P);
        std::cout << std::endl;
    }

    for ( size_t P = 19; P <= 50; ++P )
    {
        std::cout << std::setw(4)  << P;
        std::cout << std::setw(25) << "";
        std::cout << std::setw(25) << benchmark<double>(P);
        std::cout << std::endl;
    }
}

template <typename real>
real benchmark( size_t P )
{
    solid<real>              L[128], M[128];
    solid<real>              *Lptr[128], *Mptr[128];
    stopwatch<real>          clock;
    operator_handle<real>    op(P);
      buffer_handle<real>    buf(op);
    
    real x[128] {};
    real y[128] {};
    real z[128] {};

    for ( size_t i = 0; i < 128; ++i )
    {
        L[i] = solid<real>(P);
        M[i] = solid<real>(P);
        Mptr[i] = M + i;
        Lptr[i] = L + i;
        x[i] = y[i] = z[i] = 1;
    }    

    // Warm-up.
    for ( size_t i = 0; i < 128; ++i )
        m2l(op,buf,128,Mptr,Lptr,x,y,z);

    size_t trials = 1 << 8;

    // Actual benchmark
    clock.reset();
    for ( size_t i = 0; i < trials; ++i )
        m2l(op,buf,128,Mptr,Lptr,x,y,z);
    real elapsed = clock.elapsed();

    return elapsed/(trials*128);
}

