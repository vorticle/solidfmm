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
#ifndef SOLIDFMM_RANDOM_HPP
#define SOLIDFMM_RANDOM_HPP

#include <random>
#include <functional>

namespace solidfmm
{

template <typename real>
class random_real
{
public:
	random_real( real min, real max );

	real operator()() const;
private:
	std::function<real()> r;
};

template <typename real> inline
random_real<real>::random_real( real min, real max ):
 r( std::bind( std::uniform_real_distribution<>(min,max),
               std::default_random_engine() ) )
{}

template <typename real> inline
real random_real<real>::operator()() const
{
	return r();
}

}

#endif

