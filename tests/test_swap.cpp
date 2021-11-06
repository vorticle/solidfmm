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
void test_swap( size_t P )
{
    using solidfmm::dot;
    using solidfmm::solid;
    using solidfmm::random_real;
    using solidfmm::microkernel;
    using solidfmm::harmonics::R;
    using solidfmm::harmonics::S;
    using solidfmm::operator_data;
    using solidfmm::threadlocal_buffer;

    std::cout << std::scientific << std::setprecision(15);
    std::cout << std::scientific << std::setprecision(4);

    operator_data<real>         op   { std::max(P,size_t(2)) };
    threadlocal_buffer<real>    buf  { op };
    random_real<real>           rand { -1.0, 1.0 };
    const microkernel<real>  *kernel { op.kernel()  };
    const size_t                cols { kernel->cols };
    const size_t                rows { kernel->rows };

    solid<real> original( P ); 
    for ( size_t n = 0; n <  P; ++n )
    for ( size_t m = 0; m <= n; ++m )
    {
       original.re(n,m) = rand();
       original.im(n,m) = (m>0) ? rand() : 0;  
    }

    real res;
    real x = rand(), y = rand(), z = rand();

    std::cout << "M-expansion before swapping:   ";
    dot( original, S<real>( P, x, y, z ), &res );
    std::cout << std::setw(25) << res << ".\n";

    solid<real> swapped(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = original.re(n,m);
            buf.swap_imag_in[ m*cols ] = original.im(n,m);
        }

        const real *real_mat = op.real_swap_matrix(n);
        const real *imag_mat = op.imag_swap_matrix(n);
        const bool pattern { !(n&1) };
        for ( size_t m = 0; m <= n; m += rows )
        {
            kernel->swap( real_mat + (m/2)*(n+1),
                          buf.swap_real_in,
                          buf.swap_real_out + m*cols,
                          n + 1, pattern);
            kernel->swap( imag_mat + (m/2)*(n+1),
                          buf.swap_imag_in,
                          buf.swap_imag_out + m*cols,
                          n + 1, !pattern );
        }

        for ( size_t m = 0; m <= n; ++m )
        {
            swapped.re(n,m) = buf.swap_real_out[ m*cols ];
            swapped.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    real xr = z, yr = y, zr = x;

    std::cout << "M-expansion after  swapping:   ";
    dot( swapped, S<real>(P,xr,yr,zr), &res );
    std::cout << std::setw(25) << res << ".\n";

    solid<real> swapped_back(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = swapped.re(n,m);
            buf.swap_imag_in[ m*cols ] = swapped.im(n,m);
        }

        const real *real_mat = op.real_swap_matrix(n);
        const real *imag_mat = op.imag_swap_matrix(n);
        const bool pattern { !(n&1) };
        for ( size_t m = 0; m <= n; m += rows )
        {
            kernel->swap( real_mat + (m/2)*(n+1),
                          buf.swap_real_in,
                          buf.swap_real_out + m*cols,
                          n + 1, pattern);
            kernel->swap( imag_mat + (m/2)*(n+1),
                          buf.swap_imag_in,
                          buf.swap_imag_out + m*cols,
                          n + 1, !pattern );
        }

        for ( size_t m = 0; m <= n; ++m )
        {
            swapped_back.re(n,m) = buf.swap_real_out[ m*cols ];
            swapped_back.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    std::cout << "M-expansion after  reswapping: ";
    dot( swapped_back, S<real>(P,x,y,z), &res );
    std::cout << std::setw(25) << res << ".\n";
}

template <typename real>
void test_swap_transposed( size_t P )
{
    using solidfmm::dot;
    using solidfmm::solid;
    using solidfmm::random_real;
    using solidfmm::microkernel;
    using solidfmm::harmonics::R;
    using solidfmm::harmonics::S;
    using solidfmm::operator_data;
    using solidfmm::threadlocal_buffer;

    std::cout << std::scientific << std::setprecision(15);
    std::cout << std::scientific << std::setprecision(4);

    operator_data<real>         op   { std::max(P,size_t(2)) };
    threadlocal_buffer<real>    buf  { op };
    random_real<real>           rand { -1.0, 1.0 };
    const microkernel<real>  *kernel { op.kernel()  };
    const size_t                cols { kernel->cols };
    const size_t                rows { kernel->rows };

    solid<real> original( P ); 
    for ( size_t n = 0; n <  P; ++n )
    for ( size_t m = 0; m <= n; ++m )
    {
       original.re(n,m) = rand();
       original.im(n,m) = (m>0) ? rand() : 0;  
    }

    real res;
    real x = rand(), y = rand(), z = rand();


    std::cout << "L-expansion before swapping:   ";
    dot( original, R<real>( P, x, y, z ), &res );
    std::cout << std::setw(25) << res << std::endl;


    solid<real> swapped(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = original.re(n,m);
            buf.swap_imag_in[ m*cols ] = original.im(n,m);
        }

        const real *real_mat = op.real_swap_matrix_transposed(n);
        const real *imag_mat = op.imag_swap_matrix_transposed(n);
        const bool pattern { !(n&1) };
        for ( size_t m = 0; m <= n; m += rows )
        {
            kernel->swap( real_mat + (m/2)*(n+1),
                          buf.swap_real_in,
                          buf.swap_real_out + m*cols,
                          n + 1, pattern);
            kernel->swap( imag_mat + (m/2)*(n+1),
                          buf.swap_imag_in,
                          buf.swap_imag_out + m*cols,
                          n + 1, !pattern );
        }

        for ( size_t m = 0; m <= n; ++m )
        {
            swapped.re(n,m) = buf.swap_real_out[ m*cols ];
            swapped.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    real xr = z, yr = y, zr = x;

    std::cout << "L-expansion after  swapping:   ";
    dot( swapped, R<real>(P,xr,yr,zr), &res );
    std::cout << std::setw(25) << res << std::endl;

    solid<real> swapped_back(P);
    for ( size_t n = 0; n < P; ++n )
    {
        for ( size_t m = 0; m <= n; ++m )
        {
            buf.swap_real_in[ m*cols ] = swapped.re(n,m);
            buf.swap_imag_in[ m*cols ] = swapped.im(n,m);
        }

        const real *real_mat = op.real_swap_matrix_transposed(n);
        const real *imag_mat = op.imag_swap_matrix_transposed(n);
        const bool pattern { !(n&1) };
        for ( size_t m = 0; m <= n; m += rows )
        {
            kernel->swap( real_mat + (m/2)*(n+1),
                          buf.swap_real_in,
                          buf.swap_real_out + m*cols,
                          n + 1, pattern);
            kernel->swap( imag_mat + (m/2)*(n+1),
                          buf.swap_imag_in,
                          buf.swap_imag_out + m*cols,
                          n + 1, !pattern );
        }

        for ( size_t m = 0; m <= n; ++m )
        {
            swapped_back.re(n,m) = buf.swap_real_out[ m*cols ];
            swapped_back.im(n,m) = buf.swap_imag_out[ m*cols ];
        }
    }

    std::cout << "L-expansion after  reswapping: ";
    dot( swapped_back, R<real>(P,x,y,z), &res );
    std::cout << std::setw(25) << res << std::endl;
    std::cout << std::endl;
}

int main()
{
    std::cout << "Test single precision.\n";
    for ( size_t P = 1; P <= 18; ++P )
    {
        std::cout << "Order P = " << P << ".\n";
        test_swap<float>(P);
        test_swap_transposed<float>(P);
    }

    std::cout << "Test double precision.\n";
    for ( size_t P = 1; P <= 20; ++P )
    {
        std::cout << "Order P = " << P << ".\n";
        test_swap<double>(P);
        test_swap_transposed<double>(P);
    }

    return 0;
}

