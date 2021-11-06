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
#include <solidfmm/harmonics.hpp>
#include <solidfmm/stopwatch.hpp>

using namespace solidfmm;

template <typename real>
real benchmark( size_t P );

template <typename real> inline
std::complex<real> get( const solid<real> &c, int n, int m )
{
    if ( m >= 0 )
    {
        return std::complex<real> { c.re(n,m), c.im(n,m) };
    }
    else
    {
        if ( m % 2 ) return std::complex<real> { -c.re(n,-m),  c.im(n,-m) };
        else         return std::complex<real> {  c.re(n,-m), -c.im(n,-m) };
    }
}

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
    stopwatch<real>          clock;
    
    real x[128] {};
    real y[128] {};
    real z[128] {};

    for ( size_t i = 0; i < 128; ++i )
    {
        L[i] = solid<real>(P);
        M[i] = solid<real>(P);
        x[i] = y[i] = z[i] = 1;
    }    

    // Warm-up.
    for ( size_t i = 0; i < 128; ++i )
    {
        solid<real> trans = harmonics::S<real>(2*P,x[i],y[i],z[i]);

        for ( int n = 0; n < P; ++n )
        for ( int m = 0; m <=n; ++m )
        {
            std::complex<real> tmp {};
            for ( int k = 0; k < P; ++k )
            for ( int l =-k; l <=k; ++l )
            {
                tmp += conj(get(M[i],k,l)) * get(trans,n+k,m+l);
            }

            if ( n % 1 ) tmp = -tmp;
            M[i].re(n,m) += tmp.real();
            M[i].im(n,m) += tmp.imag();
        }
    }

    
    // Actual benchmark
    size_t trials = 1 << 4;
    clock.reset();
    for ( size_t t = 0; t < trials; ++t )
    for ( size_t i = 0; i < 128; ++i )
    {
        solid<real> trans = harmonics::S<real>(2*P,x[i],y[i],z[i]);

        for ( int n = 0; n < P; ++n )
        for ( int m = 0; m <=n; ++m )
        {
            std::complex<real> tmp {};
            for ( int k = 0; k < P; ++k )
            for ( int l =-k; l <=k; ++l )
            {
                tmp += conj(get(M[i],k,l)) * get(trans,n+k,m+l);
            }

            if ( n % 1 ) tmp = -tmp;
            M[i].re(n,m) += tmp.real();
            M[i].im(n,m) += tmp.imag();

        }
    }
    real elapsed = clock.elapsed();

    return elapsed/(trials*128);
}

