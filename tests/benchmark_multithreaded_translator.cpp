/*
 * Copyright (C) 2023 Matthias Kirchhart
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
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <complex>
#include <iostream>

#include <solidfmm/solid.hpp>
#include <solidfmm/stopwatch.hpp>
#include <solidfmm/translations.hpp>

using namespace solidfmm;


template <typename real>
real benchmark( size_t P, multithreaded_translator<real> &sched );

int main()
{
    std::locale loc {""};
    std::cout.imbue(loc);

    std::cout << std::setw(4)  << "P";
    std::cout << std::setw(25) << "float";
    std::cout << std::setw(25) << "double";
    std::cout << std::endl;

    multithreaded_translator<float>   float_sched(18);
    multithreaded_translator<double> double_sched(50);
    for ( size_t P = 1; P <= 18; ++P )
    {
        std::cout << std::setw(4)  << P;
        std::cout << std::setw(25) << benchmark<float >(P, float_sched);
        std::cout << std::setw(25) << benchmark<double>(P,double_sched);
        std::cout << std::endl;
    }

    for ( size_t P = 19; P <= 50; ++P )
    {
        std::cout << std::setw(4)  << P;
        std::cout << std::setw(25) << "";
        std::cout << std::setw(25) << benchmark<double>(P,double_sched);
        std::cout << std::endl;
    }
}

template <typename real>
real benchmark( size_t P, multithreaded_translator<real> &sched )
{
    size_t N = 1'000'000;

    solid<real> blank(P);
    std::vector< solid<real> > L( N, blank );
    std::vector< solid<real> > M( N, blank );
    std::vector< translation_info<real> > list(N);

    for ( size_t i = 0; i < N; ++i )
    {
        list[i].source = &M[i];
        list[i].target = &L[i];
        list[i].x = list[i].y = list[i].z = 1;
    }

    sched.m2l( list.data(), list.data() + 1024 ); // Warm up.

    stopwatch<real> clock;
    sched.m2l_unchecked( list.data(), list.data() + N );
    real runtime = clock.elapsed();

    return runtime / N;
}

