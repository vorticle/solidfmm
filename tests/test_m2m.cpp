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
#include <solidfmm/harmonics.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>
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
    using solidfmm::solid;
    using solidfmm::harmonics::R;
    using solidfmm::harmonics::S;

    real xs = 1.1, ys = 0.2, zs = 0.9;  // Source.
    real xa = 1,   ya = 0,   za = 1;    // Multipole expansion centre A.
    real xb = 0,   yb = 0,   zb = 0;    // Multipole expansion centre B
    real x  = 3.1, y  = 0.1, z  = 2.5;  // Evaluation point.

    real r = std::hypot(x-xs,y-ys);
         r = std::hypot(r   ,z-zs);

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "P = " << std::setw(2) << P << ". "
              << "Relative errors: ";
    real res = 0;
    solid<real> M = R<real>(P,xs-xa,ys-ya,zs-za);
    dot( M, S<real>(P,(x-xa),(y-ya),(z-za)), &res );
    std::cout << "M(A)-expansion: " << std::setw(10)  << r*std::abs(res-1/r) << ". ";

    solid<real> MB = R<real>(P,xs-xb,ys-yb,zs-zb);
    dot( MB, S<real>(P,(x-xb),(y-yb),(z-zb)), &res );
    std::cout << "M(B)-expansion: " << std::setw(10)  << r*std::abs(res-1/r) << ". ";

    const real xab = (xb-xa);
    const real yab = (yb-ya);
    const real zab = (zb-za);

    solid<real> M2(P);
    solidfmm::operator_data<real>      op(std::max(P,size_t(1)));
    solidfmm::threadlocal_buffer<real> buf(op);
    solid<real> *Mptr  = &M;
    solid<real> *M2ptr = &M2;
    m2m( op, buf, 1, &Mptr, &M2ptr, &xab, &yab, &zab );
    dot( M2, S<real>(P,(x-xb),(y-yb),(z-zb)), &res );
    std::cout << "M(B)-expansion after M2M: " << std::setw(10)  << r*std::abs(res-1/r) << ".\n";
}

