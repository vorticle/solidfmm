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
#ifndef SOLIDFMM_SWAP_MATRIX_HPP
#define SOLIDFMM_SWAP_MATRIX_HPP

namespace solidfmm
{

/*! 
 * \brief Wigner matrix for swapping x- and z-axis for complex valued harmonics.
 *
 * This class represents the Wigner matrices for swapping the x- and z-matrix for
 * the full, complex-valued solid harmonics. These are the matrices B_n^{m,l}
 * from Dehnen's paper, equation (64).
 */
template <typename real>
class swap_matrix
{
public:
    swap_matrix() = default;
    swap_matrix( int n );
    swap_matrix( const swap_matrix  &rhs );
    swap_matrix(       swap_matrix &&rhs ) noexcept;
    swap_matrix& operator=( const swap_matrix  &rhs );
    swap_matrix& operator=(       swap_matrix &&rhs ) noexcept;
   ~swap_matrix();

    // Returns swap Matrix entry B(n,m,l).
    // Does not check if you go out of bounds.
    real operator()( int n, int m, int l ) const noexcept;

    // Compute matrices for all orders up to but *excluding* n.
    void reserve( int n ); 

    // It is safe to call operator() for all n up to but *excluding* this number.
    int order() const noexcept; 

private:
    constexpr int idx( int n, int m, int l ) const noexcept;
    void compute_matrix( int n ) noexcept;

    // Same as operator(), but returns 0 when abs(l) > n
    real b( int n, int m, int l ) const noexcept;

private:
    int    n_end { 0 };
    real*  data  { nullptr };
};

extern template class swap_matrix<float>;
extern template class swap_matrix<double>;

}

#endif

