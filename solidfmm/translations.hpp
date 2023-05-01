/*
 * Copyright (C) 2021, 2023 Matthias Kirchhart
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
#ifndef SOLIDFMM_TRANSLATIONS_HPP
#define SOLIDFMM_TRANSLATIONS_HPP

#include <cstddef>


extern "C" { struct hwloc_topology; struct hwloc_bitmap_s; }


namespace solidfmm
{

// Forward declaration.
template <typename real> class solid;

//////////////////////////////
// Multithreaded interface. //
//////////////////////////////

template <typename real>
struct translation_info
{
    const solid<real> *source;
          solid<real> *target;
    real x, y, z;
};

template <typename real>
class multithreaded_translator
{
public:
    using item = translation_info<real>;
    using cpuset_t = hwloc_bitmap_s*;

    multithreaded_translator() = delete;
    multithreaded_translator( const multithreaded_translator  &rhs ) = delete;
    multithreaded_translator(       multithreaded_translator &&rhs ) noexcept;
    ~multithreaded_translator();
    multithreaded_translator& operator=( const multithreaded_translator  &rhs ) = delete;
    multithreaded_translator& operator=(       multithreaded_translator &&rhs ) noexcept;

    // Uses all available CPU cores.
    multithreaded_translator( size_t max_expansion_order );

    // Launches one thread per given cpuset.
    multithreaded_translator( size_t max_expansion_order,
                              hwloc_topology *topology,
                              cpuset_t *begin, cpuset_t *end );

    // Change scheduling policy. Default is "equally". Measuring makes sense
    // on heterogeneous platforms, with Performance- and Efficiency cores of
    // differing performance.
    void schedule_equally_among_threads() noexcept;
    void schedule_by_measuring_throughput();

    void m2m( const item *begin, const item *end ) const;
    void m2l( const item *begin, const item *end ) const;
    void l2l( const item *begin, const item *end ) const;

    void m2m_unchecked( const item *begin, const item *end ) const noexcept;
    void m2l_unchecked( const item *begin, const item *end ) const noexcept;
    void l2l_unchecked( const item *begin, const item *end ) const noexcept;


private:
    class impl;
    impl* p;
};

extern template class multithreaded_translator<float>;
extern template class multithreaded_translator<double>;



////////////////////////////////
// Single threaded functions. //
////////////////////////////////

// Forward declaration.
template <typename real> class operator_data;
template <typename real> class threadlocal_buffer;


void m2l( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const M, solid<float> *const *const L,
          const float *x, const float *y, const float *z );

void m2l( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const M, solid<double> *const *const L,
          const double *x, const double *y, const double *z );

void m2l( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          const translation_info<float> *begin, const translation_info<float> *end );

void m2l( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          const translation_info<double> *begin, const translation_info<double> *end );




void m2m( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const Min, solid<float> *const *const Mout,
          const float *x, const float *y, const float *z );

void m2m( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const Min, solid<double> *const *const Mout,
          const double *x, const double *y, const double *z );

void m2m( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          const translation_info<float> *begin, const translation_info<float> *end );

void m2m( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          const translation_info<double> *begin, const translation_info<double> *end );




void l2l( const operator_data<float> &op, threadlocal_buffer<float> &buf, 
          size_t howmany, const solid<float> *const *const Min, solid<float> *const *const Mout,
          const float *x, const float *y, const float *z );

void l2l( const operator_data<double> &op, threadlocal_buffer<double> &buf, 
          size_t howmany, const solid<double> *const *const Lin, solid<double> *const *const Lout,
          const double *x, const double *y, const double *z );

void l2l( const operator_data<float> &op, threadlocal_buffer<float> &buf,
          const translation_info<float> *begin, const translation_info<float> *end );

void l2l( const operator_data<double> &op, threadlocal_buffer<double> &buf,
          const translation_info<double> *begin, const translation_info<double> *end );




void m2l_unchecked( const operator_data<float> &op, threadlocal_buffer<float> &buf,
                    const translation_info<float> *begin, const translation_info<float> *end ) noexcept;

void m2l_unchecked( const operator_data<double> &op, threadlocal_buffer<double> &buf,
                    const translation_info<double> *begin, const translation_info<double> *end ) noexcept;

void m2m_unchecked( const operator_data<float> &op, threadlocal_buffer<float> &buf,
                    const translation_info<float> *begin, const translation_info<float> *end ) noexcept;

void m2m_unchecked( const operator_data<double> &op, threadlocal_buffer<double> &buf,
                    const translation_info<double> *begin, const translation_info<double> *end ) noexcept;

void l2l_unchecked( const operator_data<float> &op, threadlocal_buffer<float> &buf,
                    const translation_info<float> *begin, const translation_info<float> *end ) noexcept;

void l2l_unchecked( const operator_data<double> &op, threadlocal_buffer<double> &buf,
                    const translation_info<double> *begin, const translation_info<double> *end ) noexcept;

}

#endif

