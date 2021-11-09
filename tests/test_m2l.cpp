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

#include <solidfmm/solid.hpp>
#include <solidfmm/handles.hpp>
#include <solidfmm/harmonics.hpp>
#include <solidfmm/translations.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

template <typename real> void test( size_t P );

int main()
{
    std::cout << "Testing local and multipole expansions.\n";
    std::cout << "Testing single precision.\n";
    for ( size_t P = 0; P <= 18; ++P )
        test<float>(P);
    std::cout << std::endl;

    std::cout << "Testing double precision.\n";
    for ( size_t P = 0; P <= 40; ++P )
        test<double>(P);
    std::cout << std::endl;
}

template <typename real>
void test( size_t P )
{
    using solidfmm::m2l;
    using solidfmm::solid;
    using solidfmm::harmonics::R;
    using solidfmm::harmonics::S;
    using solidfmm::operator_handle;
    using solidfmm::  buffer_handle;

    real xs = 1.1, ys = 0.2, zs = 0.9;  // Source.
    real xa = 1,   ya = 0,   za = 1;    // Multipole expansion centre.
    real xb = 0.0, yb = 0.0, zb = 2;    // Local expansion centre.
    real x  = 0.1, y  = 0.1, z  = 2.5;  // Evaluation point.

    real r = std::hypot(x-xs,y-ys);
         r = std::hypot(r   ,z-zs);

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "P = " << std::setw(2) << P << ". "
              << "Relative errors: ";

    real res = 0;
    solid<real> M = R<real>(P,xs-xa,ys-ya,zs-za);
    dot( M, S<real>(P,(x-xa),(y-ya),(z-za)), &res );
    std::cout << "M-expansion: " << std::setw(10)  << r*std::abs(res-1/r) << ". ";

    solid<real> L = S<real>(P,xs-xb,ys-yb,zs-zb);
    dot( L, R<real>(P,x-xb,y-yb,z-zb), &res );
    std::cout << "L-expansion: " << std::setw(10)  << r*std::abs(res-1/r) << ". ";

    const real xab = (xb-xa);
    const real yab = (yb-ya);
    const real zab = (zb-za);

    L.resize(P); 
    L.zeros(); 

    operator_handle<real> op(std::max(P,size_t(1)));
      buffer_handle<real> buf(op);

    solid<real> *Mptr[] = { &M, &M, &M, &M, &M, &M, &M, &M,
                            &M, &M, &M, &M, &M, &M, &M, &M,
                            &M, &M, &M, &M, &M, &M, &M, &M,
                            &M, &M, &M, &M, &M, &M, &M, &M };
    solid<real> *Lptr[] = { &L, &L, &L, &L, &L, &L, &L, &L,
                            &L, &L, &L, &L, &L, &L, &L, &L,
                            &L, &L, &L, &L, &L, &L, &L, &L,
                            &L, &L, &L, &L, &L, &L, &L, &L };
    const real   xx[]   = { xab, xab, xab, xab, xab, xab, xab, xab,
                            xab, xab, xab, xab, xab, xab, xab, xab,
                            xab, xab, xab, xab, xab, xab, xab, xab,
                            xab, xab, xab, xab, xab, xab, xab, xab };
    const real   yy[]   = { yab, yab, yab, yab, yab, yab, yab, yab,
                            yab, yab, yab, yab, yab, yab, yab, yab,
                            yab, yab, yab, yab, yab, yab, yab, yab,
                            yab, yab, yab, yab, yab, yab, yab, yab };
    const real   zz[]   = { zab, zab, zab, zab, zab, zab, zab, zab,
                            zab, zab, zab, zab, zab, zab, zab, zab,
                            zab, zab, zab, zab, zab, zab, zab, zab,
                            zab, zab, zab, zab, zab, zab, zab, zab };

    m2l( op, buf, 32, Mptr, Lptr, xx, yy, zz );


    dot( L, R<real>(P, (x-xb), (y-yb), (z-zb)), &res ); res /= 32;
    std::cout << "L-expansion after M2L: " << std::setw(10)  << r*std::abs(res-1/r) << ".\n";
}

