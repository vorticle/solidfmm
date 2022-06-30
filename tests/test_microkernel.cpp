/*
 * Copyright (C) 2022 Matthias Kirchhart
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
#include <solidfmm/random.hpp>
#include <solidfmm/microkernel_test.hpp>
#include <solidfmm/threadlocal_buffer.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <iostream>


///////////////////////////////////////////////////
// You only need to modify the code block below. //
///////////////////////////////////////////////////

#include <solidfmm/microkernel_generic.hpp> // TODO: Include YOUR microkernel here.

namespace solidfmm
{

// TODO: Write the floating point type you want to test here.
using T = float;

// TODO: Have this function return the kernel you want to test.
microkernel<T>* make_my_kernel() 
{
    return new microkernel_float_generic;
}

}

//////////////////////////////////////////////////////
// ! No need to modify anything beyond this point ! //
//////////////////////////////////////////////////////



namespace solidfmm
{

microkernel<T>* make_reference_kernel( microkernel<T> *my_kernel )
{
    return new microkernel_test<T> { my_kernel->rows,
                                     my_kernel->cols,
                                     my_kernel->alignment };
}


// Performs zm2l on random data and compares the results.
void test_zm2l( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    T fac[ 100 ];

    std::cout << "Testing zm2l..."; std::cout.flush();

    for ( size_t i = 0; i < 100; ++i )
        fac[ i ] = rand();

    for ( size_t i = 0; i < 50; ++i )
    for ( size_t j = 0; j < my_kernel->cols; ++j )
    {
        T r = rand();
        my_buf.trans_real_in[0][ i*cols + j ] = r;
       ref_buf.trans_real_in[0][ i*cols + j ] = r;

        r = rand();
        my_buf.trans_imag_in[0][ i*cols + j ] = r;
       ref_buf.trans_imag_in[0][ i*cols + j ] = r;
    }

     my_kernel->zm2l( fac,  my_buf.trans_real_in[0],  my_buf.trans_real_out[0], 50, false );
    ref_kernel->zm2l( fac, ref_buf.trans_real_in[0], ref_buf.trans_real_out[0], 50, false );
     my_kernel->zm2l( fac,  my_buf.trans_imag_in[0],  my_buf.trans_imag_out[0], 50, true  );
    ref_kernel->zm2l( fac, ref_buf.trans_imag_in[0], ref_buf.trans_imag_out[0], 50, true  );
   
    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.trans_real_out[0][ i*cols + j ];
        T res1 = ref_buf.trans_real_out[0][ i*cols + j ];
        T res2 =  my_buf.trans_imag_out[0][ i*cols + j ];
        T res3 = ref_buf.trans_imag_out[0][ i*cols + j ];

        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing zm2l with pattern = false.";
            return;
        }

        if ( abs((res2-res3)/res3) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res2 = " << res2 << ", res3 = " << res3 << std::endl;
            std::cerr << "abs_error " << abs(res2-res3) << ", rel_error = " << abs((res2-res3)/res3) << std::endl;
            std::cerr << "Error when testing zm2l with pattern = true.";
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}

// Performs zm2m on random data and compares the results.
void test_zm2m( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    T fac[ 100 ];

    std::cout << "Testing zm2m..."; std::cout.flush();

    for ( size_t i = 0; i < 100; ++i )
        fac[ i ] = rand();

    for ( size_t i = 0; i < 50; ++i )
    for ( size_t j = 0; j < my_kernel->cols; ++j )
    {
        T r = rand();
        my_buf.trans_real_in[0][ i*cols + j ] = r;
       ref_buf.trans_real_in[0][ i*cols + j ] = r;

        r = rand();
        my_buf.trans_imag_in[0][ i*cols + j ] = r;
       ref_buf.trans_imag_in[0][ i*cols + j ] = r;
    }
 
     my_kernel->zm2m( fac,  my_buf.trans_real_in[0],  my_buf.trans_real_out[0], 50 );
    ref_kernel->zm2m( fac, ref_buf.trans_real_in[0], ref_buf.trans_real_out[0], 50 );
   
    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.trans_real_out[0][ i*cols + j ];
        T res1 = ref_buf.trans_real_out[0][ i*cols + j ];

        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing zm2m.";
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}

// Performs swap on random data and compares the results.
void test_swap( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::unique_ptr<T[]> swap_mat { new T[ (rows/2)*50 ] };
    for ( size_t i = 0; i < (rows/2)*50; ++i )
        swap_mat[ i ] = rand();
    

    std::cout << "Testing swap..."; std::cout.flush();


    for ( size_t i = 0; i < 50; ++i )
    for ( size_t j = 0; j < my_kernel->cols; ++j )
    {
        T r = rand();
        my_buf.swap_real_in[ i*cols + j ] = r;
       ref_buf.swap_real_in[ i*cols + j ] = r;

        r = rand();
        my_buf.swap_imag_in[ i*cols + j ] = r;
       ref_buf.swap_imag_in[ i*cols + j ] = r;
    }
 
     my_kernel->swap( swap_mat.get(),  my_buf.swap_real_in,  my_buf.swap_real_out, 50, false );
    ref_kernel->swap( swap_mat.get(), ref_buf.swap_real_in, ref_buf.swap_real_out, 50, false );
     my_kernel->swap( swap_mat.get(),  my_buf.swap_imag_in,  my_buf.swap_imag_out, 50, true );
    ref_kernel->swap( swap_mat.get(), ref_buf.swap_imag_in, ref_buf.swap_imag_out, 50, true );
   
    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.swap_real_out[ i*cols + j ];
        T res1 = ref_buf.swap_real_out[ i*cols + j ];
        T res2 =  my_buf.swap_imag_out[ i*cols + j ];
        T res3 = ref_buf.swap_imag_out[ i*cols + j ];

        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing swap with pattern == false and k even." << std::endl;
            return;
        }

        if ( abs((res2-res3)/res3) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res2 = " << res2 << ", res3 = " << res3 << std::endl;
            std::cerr << "abs_error " << abs(res2-res3) << ", rel_error = " << abs((res2-res3)/res3) << std::endl;
            std::cerr << "Error when testing swap with pattern == true and k even." << std::endl;
            return;
        }
    }

    // Testing k odd.
     my_kernel->swap( swap_mat.get(),  my_buf.swap_real_in,  my_buf.swap_real_out, 49, false );
    ref_kernel->swap( swap_mat.get(), ref_buf.swap_real_in, ref_buf.swap_real_out, 49, false );
     my_kernel->swap( swap_mat.get(),  my_buf.swap_imag_in,  my_buf.swap_imag_out, 49, true );
    ref_kernel->swap( swap_mat.get(), ref_buf.swap_imag_in, ref_buf.swap_imag_out, 49, true );
   
    for ( size_t i = 0; i < rows; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.swap_real_out[ i*cols + j ];
        T res1 = ref_buf.swap_real_out[ i*cols + j ];
        T res2 =  my_buf.swap_imag_out[ i*cols + j ];
        T res3 = ref_buf.swap_imag_out[ i*cols + j ];

        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing swap with pattern == false and k odd." << std::endl;
            return;
        }

        if ( abs((res2-res3)/res3) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res2 = " << res2 << ", res3 = " << res3 << std::endl;
            std::cerr << "abs_error " << abs(res2-res3) << ", rel_error = " << abs((res2-res3)/res3) << std::endl;
            std::cerr << "Error when testing swap with pattern == true and k odd." << std::endl;
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}


// Computes the "Euler data" on random shift vectors and compares the results
// Also checks for border cases with zero shift.
void test_euler( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                 threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing euler..."; std::cout.flush();

    for ( size_t i = 0; i < cols; ++i )
    {
        my_buf.x[i] = ref_buf.x[i] = rand();
        my_buf.y[i] = ref_buf.y[i] = rand();
        my_buf.z[i] = ref_buf.z[i] = rand();
    }

     my_kernel->euler(  my_buf.x,          my_buf.y, my_buf.z, my_buf.r, my_buf.rinv,
                        my_buf.cos_alpha,  my_buf.sin_alpha,
                        my_buf.cos_beta,   my_buf.sin_beta, 50 );
    ref_kernel->euler( ref_buf.x,         ref_buf.y, ref_buf.z, ref_buf.r, ref_buf.rinv,
                       ref_buf.cos_alpha, ref_buf.sin_alpha,
                       ref_buf.cos_beta,  ref_buf.sin_beta, 50 );

    for ( size_t i = 0; i < 50;   ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.r[ i*cols + j ];
        T res1 = ref_buf.r[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing r result of euler." << std::endl;
            return;
        }

        res0 =  my_buf.rinv[ i*cols + j ];
        res1 = ref_buf.rinv[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing rinv result of euler." << std::endl;
            return;
        }

        res0 =  my_buf.cos_alpha[ i*cols + j ];
        res1 = ref_buf.cos_alpha[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing cos_alpha result of euler." << std::endl;
            return;
        }

        res0 =  my_buf.sin_alpha[ i*cols + j ];
        res1 = ref_buf.sin_alpha[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing sin_alpha result of euler." << std::endl;
            return;
        }

        res0 =  my_buf.cos_beta[ i*cols + j ];
        res1 = ref_buf.cos_beta[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing cos_beta result of euler." << std::endl;
            return;
        }

        res0 =  my_buf.sin_beta[ i*cols + j ];
        res1 = ref_buf.sin_beta[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing sin_beta result of euler." << std::endl;
            return;
        }
    }


    // Test if zero shift is handled correctly.
    for ( size_t i = 0; i < cols; ++i )
    {
        my_buf.x[i] = ref_buf.x[i] = 0;
        my_buf.y[i] = ref_buf.y[i] = 0;
        my_buf.z[i] = ref_buf.z[i] = 0;
    }

     my_kernel->euler(  my_buf.x,          my_buf.y, my_buf.z, my_buf.r, my_buf.rinv,
                        my_buf.cos_alpha,  my_buf.sin_alpha,
                        my_buf.cos_beta,   my_buf.sin_beta, 50 );
    ref_kernel->euler( ref_buf.x,         ref_buf.y, ref_buf.z, ref_buf.r, ref_buf.rinv,
                       ref_buf.cos_alpha, ref_buf.sin_alpha,
                       ref_buf.cos_beta,  ref_buf.sin_beta, 50 );

    for ( size_t i = 0; i < 50;   ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.r[ i*cols + j ];
        T res1 = ref_buf.r[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing r result of euler and zero shift." << std::endl;
            return;
        }

        res0 =  my_buf.rinv[ i*cols + j ];
        res1 = ref_buf.rinv[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing rinv result of euler and zero shift." << std::endl;
            return;
        }

        res0 =  my_buf.cos_alpha[ i*cols + j ];
        res1 = ref_buf.cos_alpha[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing cos_alpha result of euler and zero shift." << std::endl;
            return;
        }

        res0 =  my_buf.sin_alpha[ i*cols + j ];
        res1 = ref_buf.sin_alpha[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing sin_alpha result of euler and zero shift." << std::endl;
            return;
        }

        res0 =  my_buf.cos_beta[ i*cols + j ];
        res1 = ref_buf.cos_beta[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing cos_beta result of euler and zero shift." << std::endl;
            return;
        }

        res0 =  my_buf.sin_beta[ i*cols + j ];
        res1 = ref_buf.sin_beta[ i*cols + j ];
        if ( abs((res0-res1)) > 128*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing sin_beta result of euler and zero shift." << std::endl;
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}

// Test rotation and scaling on random shift vectors and compares the results.
void test_rotscale( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                    threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing rotscale..."; std::cout.flush();

    // Generate random rotations.
    for ( size_t i = 0; i < cols; ++i )
    {
        ref_buf.x[i] = rand();
        ref_buf.y[i] = rand();
        ref_buf.z[i] = rand();
    }
    ref_kernel->euler( ref_buf.x,         ref_buf.y, ref_buf.z, ref_buf.r, ref_buf.rinv,
                       ref_buf.cos_alpha, ref_buf.sin_alpha,
                       ref_buf.cos_beta,  ref_buf.sin_beta, 50 );

    std::unique_ptr<T, decltype(&std::free) > scale { (T*) std::aligned_alloc( sizeof(T)*cols, my_kernel->alignment), &std::free };
    if ( ! scale ) throw std::bad_alloc {};
    for ( size_t i = 0; i < cols; ++i )
        scale.get()[ i ] = rand();


    for ( size_t i = 0; i < 50;   ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        my_buf.swap_real_in[ i*cols + j ] = ref_buf.swap_real_in[ i*cols + j ] = rand();
        my_buf.swap_imag_in[ i*cols + j ] = ref_buf.swap_imag_in[ i*cols + j ] = rand();
    }

    my_kernel->rotscale( ref_buf.cos_alpha, ref_buf.sin_alpha, scale.get(),
                          my_buf.swap_real_in,  my_buf.swap_imag_in,
                          my_buf.swap_real_out, my_buf.swap_imag_out, 50, true );

    ref_kernel->rotscale( ref_buf.cos_alpha, ref_buf.sin_alpha, scale.get(),
                          ref_buf.swap_real_in,  ref_buf.swap_imag_in,
                          ref_buf.swap_real_out, ref_buf.swap_imag_out, 50, true );

    for ( size_t i = 0; i < 50; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.swap_real_out[ i*cols + j ];
        T res1 = ref_buf.swap_real_out[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing real part of result of rotscale with forward = true." << std::endl;
            return;
        }

        res0 =  my_buf.swap_imag_out[ i*cols + j ];
        res1 = ref_buf.swap_imag_out[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing imaginary part of result of rotscale with forward = true." << std::endl;
            return;
        }
    }

    my_kernel->rotscale( ref_buf.cos_alpha, ref_buf.sin_alpha, scale.get(),
                          my_buf.swap_real_in,  my_buf.swap_imag_in,
                          my_buf.swap_real_out, my_buf.swap_imag_out, 50, false );

    ref_kernel->rotscale( ref_buf.cos_alpha, ref_buf.sin_alpha, scale.get(),
                          ref_buf.swap_real_in,  ref_buf.swap_imag_in,
                          ref_buf.swap_real_out, ref_buf.swap_imag_out, 50, false );

    for ( size_t i = 0; i < 50; ++i )
    for ( size_t j = 0; j < cols; ++j )
    {
        T res0 =  my_buf.swap_real_out[ i*cols + j ];
        T res1 = ref_buf.swap_real_out[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing real part of result of rotscale with forward = false." << std::endl;
            return;
        }

        res0 =  my_buf.swap_imag_out[ i*cols + j ];
        res1 = ref_buf.swap_imag_out[ i*cols + j ];
        if ( abs((res0-res1)/res1) > 1024*std::numeric_limits<T>::epsilon() )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "i = " << i << ", j = " << j << std::endl;
            std::cerr << "res0 = " << res0 << ", res1 = " << res1 << std::endl;
            std::cerr << "abs_error " << abs(res0-res1) << ", rel_error = " << abs((res0-res1)/res1) << std::endl;
            std::cerr << "Error when testing imaginary part of result of rotscale with forward = false." << std::endl;
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}

// Checks if the copy operations work correctly.
void test_swap2trans( microkernel<T> *my_kernel, microkernel<T>*,
                      threadlocal_buffer<T> &my_buf, threadlocal_buffer<T>& )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing swap2trans_buf..."; std::cout.flush();

    const size_t n = 49;
    for ( size_t m = 0; m <= n;   ++m )
    for ( size_t j = 0; j < cols; ++j )
    {
        my_buf.swap_real_in[ m*cols + j ] = rand();
        my_buf.swap_imag_in[ m*cols + j ] = rand();
    }

    my_kernel->swap2trans_buf( my_buf.swap_real_in, my_buf.swap_imag_in,
                               my_buf.trans_real_in, my_buf.trans_imag_in, n );

    for ( size_t m = 0; m <= n;   ++m )
    for ( size_t j = 0; j < cols; ++j )
    {
        if ( my_buf.swap_real_in[ m*cols + j ] !=
             my_buf.trans_real_in[ m ][ (n-m)*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Real part n = 49, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }

        if ( my_buf.swap_imag_in[ m*cols + j ] !=
             my_buf.trans_imag_in[ m ][ (n-m)*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Imaginary part n = 49, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }
    }

    std::cout << "PASSED." << std::endl;
}

// Checks if the copy operations work correctly.
void test_trans2swap( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                      threadlocal_buffer<T> &my_buf, threadlocal_buffer<T>& )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing trans2swap_buf..."; std::cout.flush();

    const size_t n = 49;
    for ( size_t m = 0; m <= n;   ++m )
    for ( size_t j = 0; j < cols; ++j )
    {
        my_buf.trans_real_in[m][ (n-m)*cols + j ] = rand();
        my_buf.trans_imag_in[m][ (n-m)*cols + j ] = rand();
    }

    my_kernel->trans2swap_buf( my_buf.trans_real_in, my_buf.trans_imag_in,
                               my_buf.swap_real_in, my_buf.swap_imag_in, n, 25 );

    for ( size_t m = 0; m < 25;   ++m )
    for ( size_t j = 0; j < cols; ++j )
    {
        if ( my_buf.swap_real_in[ m*cols + j ] !=
             my_buf.trans_real_in[ m ][ (n-m)*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Real part n = 50, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }

        if ( my_buf.swap_imag_in[ m*cols + j ] !=
             my_buf.trans_imag_in[ m ][ (n-m)*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Imaginary part n = 50, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }
    }

    for ( size_t m = 25; m <= n;   ++m )
    for ( size_t j =  0; j < cols; ++j )
    {
        if ( my_buf.swap_real_in[ m*cols + j ] != 0 )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Real part n = 49, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }

        if ( my_buf.swap_imag_in[ m*cols + j ] != 0 )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "Imaginary part n = 49, m = " << m << "." << std::endl;
            std::cerr << "Error when testing swap2trans_buf." << std::endl;
            return;
        }
    }


    std::cout << "PASSED." << std::endl;
}

// Checks if the copy operations work correctly.
void test_solid2swap( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                      threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing solid2buf..."; std::cout.flush();

    std::vector< solid<T>  > solids( cols, solid<T>(50) );

    for ( size_t i = 0; i < cols; ++i )
    {
        my_buf.P_in[ i ]      = ref_buf.P_in[ i ] = 50;
        my_buf.solid_in[ i ]  = ref_buf.solid_in[ i ] = solids[i].memptr();
    }
    my_buf.P_in[ 0 ] = ref_buf.P_in[ 0 ] = 2; // Test if zeros are created when n > P.

    const size_t n = 49;
    for ( size_t m = 0; m <= n; ++m )
    for ( size_t i = 0; i <  cols; ++i )
    {
        solids[i].re(n,m) = rand();
        solids[i].im(n,m) = rand();
    }

    my_kernel->solid2buf( my_buf.solid_in, my_buf.zeros, my_buf.P_in,
                          my_buf.swap_real_in, my_buf.swap_imag_in, n );

    ref_kernel->solid2buf( ref_buf.solid_in, ref_buf.zeros, ref_buf.P_in,
                           ref_buf.swap_real_in, ref_buf.swap_imag_in, n );

    for ( size_t m = 0; m <= n;   ++m )
    for ( size_t j = 0; j < cols; ++j )
    {
        if ( ref_buf.swap_real_in[ m*cols + j ] !=  
              my_buf.swap_real_in[ m*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "m = " << m << ", j = " << j << ", real part.\n";
            std::cerr << "Error when testing solid2buf." << std::endl;
            return;
        }

        if ( ref_buf.swap_imag_in[ m*cols + j ] !=  
              my_buf.swap_imag_in[ m*cols + j ] )
        {
            std::cout << "FAILED!" << std::endl;
            std::cerr << "m = " << m << ", j = " << j << ", real part.\n";
            std::cerr << "Error when testing solid2buf." << std::endl;
            return;
        }
    }    

    std::cout << "PASSED." << std::endl;
}

// Checks if the copy operations work correctly.
void test_swap2solid( microkernel<T> *my_kernel, microkernel<T> *ref_kernel,
                     threadlocal_buffer<T> &my_buf, threadlocal_buffer<T> &ref_buf )
{
    using std::abs;
    const size_t rows = my_kernel->rows;
    const size_t cols = my_kernel->cols;
    random_real<T> rand(-1,1);

    std::cout << "Testing buf2solid..."; std::cout.flush();
    std::vector< solid<T>  >  my_solids( cols, solid<T>(50) );
    std::vector< solid<T>  > ref_solids( cols, solid<T>(50) );

     my_solids[ 0 ].reinit(2);
    ref_solids[ 0 ].reinit(2);

    for ( size_t i = 0; i < cols; ++i )
    {
         my_buf.P_out[ i ]     = ref_buf.P_out[ i ] = 50;
         my_buf.solid_out[ i ] =  my_solids[i].memptr();
        ref_buf.solid_out[ i ] = ref_solids[i].memptr();
    }
    my_buf.P_out[ 0 ] = ref_buf.P_out[ 0 ] = 2;      // Test if we accitendially write beyond our memory limits.

    if ( cols > 2 ) // Test if accumulation works.
    {
         my_buf.solid_out[ 2 ] =  my_solids[1].memptr();  
        ref_buf.solid_out[ 2 ] = ref_solids[1].memptr(); 
    }

    for ( size_t m = 0; m < 50; ++m )
    for ( size_t i = 0; i < cols; ++i )
    {
        my_buf.swap_real_in[ m*cols + i ] = ref_buf.swap_real_in[ m*cols + i ] = rand();
        my_buf.swap_imag_in[ m*cols + i ] = ref_buf.swap_imag_in[ m*cols + i ] = rand();
    }

    my_kernel->buf2solid( my_buf.swap_real_in, my_buf.swap_imag_in, my_buf.solid_out,
                          my_buf.trash, my_buf.P_out, 49 );

    ref_kernel->buf2solid( ref_buf.swap_real_in, ref_buf.swap_imag_in, ref_buf.solid_out,
                           ref_buf.trash, ref_buf.P_out, 49 );

    for ( size_t i = 0; i < cols; ++i )
    for ( size_t n = 0; n < my_solids[i].order(); ++n )
    for ( size_t m = 0; m <= n;   ++m )
    {
        if ( my_solids[i].re(n,m) != ref_solids[i].re(n,m) )
        {
            std::cout << "FAILED." << std::endl;
            std::cerr << "Real part of i = " << i << ", n = " << n << "m = " << m << '\n';
            std::cerr << "Error when testing swap2solid.\n";
            return;
        }

        if ( my_solids[i].im(n,m) != ref_solids[i].im(n,m) )
        {
            std::cout << "FAILED." << std::endl;
            std::cerr << "Imaginary part of i = " << i << ", n = " << n << "m m = " << m << '\n';
            std::cerr << "Error when testing swap2solid.\n";
            return;
        }
    }

    std::cout << "PASSED." << std::endl;

}

void test()
{
    std::unique_ptr<microkernel<T>>   my_kernel { make_my_kernel() };
    std::unique_ptr<microkernel<T>>  ref_kernel { make_reference_kernel( my_kernel.get() ) };

    threadlocal_buffer<T>  my_buf( 50,  my_kernel.get() );
    threadlocal_buffer<T> ref_buf( 50, ref_kernel.get() );

    test_zm2l       ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_zm2m       ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_swap       ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_euler      ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_rotscale   ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_swap2trans ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_trans2swap ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_solid2swap ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
    test_swap2solid ( my_kernel.get(), ref_kernel.get(), my_buf, ref_buf );
}

}

int main()
{
    solidfmm::test();
}

