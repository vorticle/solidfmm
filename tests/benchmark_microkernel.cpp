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
#include <memory>
#include <cstdlib>
#include <iostream>

#include <solidfmm/stopwatch.hpp>
#include <solidfmm/microkernel.hpp>
#include <solidfmm/operator_data.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

using namespace solidfmm;

template <typename real>
real swap_benchmark( size_t P );

template <typename real>
real zm2l_benchmark( size_t P );

int main()
{
    std::cout << "Swapping:\n";
    std::cout << "GFLOP/s for single precision: " << swap_benchmark<float >(18)  << ".\n";
    std::cout << "GFLOP/s for double precision: " << swap_benchmark<double>(40)  << ".\n";

    std::cout << std::endl;

    std::cout << "z-Transition:\n";
    std::cout << "GFLOP/s for single precision: " << zm2l_benchmark<float >(18)  << ".\n";
    std::cout << "GFLOP/s for double precision: " << zm2l_benchmark<double>(40)  << ".\n";
}

template <typename real>
real swap_benchmark( size_t P )
{
    stopwatch<real>          clock;
    operator_data<real>      op (P);
    threadlocal_buffer<real> buf(op);
    const microkernel<real> *kernel { op.kernel() };

    const real* mat = op.real_swap_matrix( P-1 );

    // Warm-up.
    for ( size_t i = 0; i < 128; ++i )
        kernel->swap( mat, buf.swap_real_in, buf.swap_real_out, P, false );

    size_t trials = 1 << 24;

    // Actual benchmark
    clock.reset();
    for ( size_t i = 0; i < trials; ++i )
        kernel->swap( mat, buf.swap_real_in, buf.swap_real_out, P, false );
    real elapsed = clock.elapsed();

    // Flop-count: 2*m*n*k / 2
    // Divided by two because only half of the entries are non-zero
    size_t flops = trials * kernel->rows * kernel->cols * P;

    real rflops = static_cast<real>(flops);
    real giga   = static_cast<real>(1e-9);
   
    return giga*rflops/elapsed;
}

template <typename real>
real zm2l_benchmark( size_t P )
{
    stopwatch<real>          clock;
    operator_data<real>      op (P);
    threadlocal_buffer<real> buf(op);
    const microkernel<real> *kernel { op.kernel() };

    // Warm-up.
    for ( size_t i = 0; i < 128; ++i )
        kernel->zm2l( op.faculties(), buf.trans_real_in[0], buf.trans_real_out[0], P, false );

    size_t trials = 1 << 24;
    clock.reset();
    for ( size_t i = 0; i < trials; ++i )
        kernel->zm2l( op.faculties(), buf.trans_real_in[0], buf.trans_real_out[0], P, false );
    real elapsed = clock.elapsed();

    // Flop-count per trial: 2*m*n*k
    size_t flops = trials * kernel->rows * kernel->cols * P * 2;

    real rflops = static_cast<real>(flops);
    real giga   = static_cast<real>(1e-9);
   
    return giga*rflops/elapsed;
}

