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

#include <cmath>
#include <iomanip>
#include <iostream>

#include <solidfmm/solid.hpp>
#include <solidfmm/random.hpp>
#include <solidfmm/harmonics.hpp>
#include <solidfmm/microkernel.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

template <typename real>
void test_rotation( size_t P )
{
    using solidfmm::dot;
    using solidfmm::solid;
    using solidfmm::random_real;
    using solidfmm::microkernel;
    using solidfmm::harmonics::R;
    using solidfmm::harmonics::S;
    using solidfmm::operator_data;
    using solidfmm::threadlocal_buffer;

    std::cout << "Order P = " << P << ".\n";
    std::cout << std::scientific << std::setprecision(15);

    operator_data<real>         op   { std::max(P,size_t(2)) };
    threadlocal_buffer<real>    buf  { op };
    random_real<real>           rand { -1.0, 1.0 };
    const microkernel<real>  *kernel { op.kernel()  };
    const size_t                cols { kernel->cols };

    solid<real> original( P );
    for ( size_t n = 0; n <  P; ++n )
    for ( size_t m = 0; m <= n; ++m )
    {
        original.re(n,m) = rand();
        original.im(n,m) = (m>0) ? rand() : 0;  
    }

    real res;
    real x = rand(), y = rand(), z = rand();

    std::cout << "M-expansion: ";
    dot( original, S<real>( P, x, y, z ), &res );
    std::cout << std::setw(25) << res << ", ";

    std::cout << "L-expansion: ";
    dot( original, R<real>( P, x, y, z ), &res );
    std::cout << std::setw(25) << res << std::endl;

    constexpr real pi = 3.14159265358979323846264338327950288;
    const real alpha = rand()*pi;
    const real c     = std::cos(alpha);
    const real s     = std::sin(alpha);

    for ( size_t n = 0; n < P; ++n )
        buf.r[ n*cols ] = 1;

    buf.cos_alpha[ 0*cols ] = 1; buf.sin_alpha[ 0*cols ] = 0;
    buf.cos_alpha[ 1*cols ] = c; buf.sin_alpha[ 1*cols ] = s;
    for ( size_t n = 2; n <= P; ++n )
    {
        buf.cos_alpha[ n*cols ] = c*buf.cos_alpha[ (n-1)*cols ] - s*buf.sin_alpha[ (n-1)*cols ];
        buf.sin_alpha[ n*cols ] = s*buf.cos_alpha[ (n-1)*cols ] + c*buf.sin_alpha[ (n-1)*cols ];
    }

    solid<real> rotated(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = original.re(n,m);
            buf.swap_imag_in[ m*cols ] = original.im(n,m);
        }

        kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.r,
                          buf.swap_real_in , buf.swap_imag_in ,
                          buf.swap_real_out, buf.swap_imag_out, n + 1, true );

        for ( size_t m = 0; m <= n; ++m )
        {
            rotated.re(n,m) = buf.swap_real_out[ m*cols ];
            rotated.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    real xr = c*x - s*y, yr = s*x + c*y, zr = z;

    std::cout << "M-expansion: ";
    dot( rotated, S<real>(P,xr,yr,zr), &res );
    std::cout << std::setw(25) << res << ", ";

    std::cout << "L-expansion: ";
    dot( rotated, R<real>(P,xr,yr,zr), &res );
    std::cout << std::setw(25) << res << std::endl;

    solid<real> rotated_back(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = rotated.re(n,m);
            buf.swap_imag_in[ m*cols ] = rotated.im(n,m);
        }

        kernel->rotscale( buf.cos_alpha, buf.sin_alpha, buf.r,
                          buf.swap_real_in , buf.swap_imag_in ,
                          buf.swap_real_out, buf.swap_imag_out, n + 1, false );

        for ( size_t m = 0; m <= n; ++m )
        {
            rotated_back.re(n,m) = buf.swap_real_out[ m*cols ];
            rotated_back.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    std::cout << "M-expansion: ";
    dot( rotated_back, S<real>(P,x,y,z), &res );
    std::cout << std::setw(25) << res << ", ";

    std::cout << "L-expansion: ";
    dot( rotated_back, R<real>(P,x,y,z), &res );
    std::cout << std::setw(25) << res << std::endl;
    std::cout << std::endl;
}

int main()
{
    std::cout << "Test single precision.\n";
    for ( size_t i = 1; i <= 10; ++i )
        test_rotation<float>(i);

    std::cout << "Test double precision.\n";
    for ( size_t i = 1; i <= 10; ++i )
        test_rotation<double>(i);

    return 0;
}

