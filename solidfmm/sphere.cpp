/*
 * Copyright (C) 2023 Matthias Kirchhart
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
#include <solidfmm/sphere.hpp>

#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace solidfmm
{

// Explicit template instantiations.
template struct sphere<float >;
template struct sphere<double>;

namespace // Anonymous.
{

template <typename real>
constexpr real eps() noexcept
{
    return 16*std::numeric_limits<real>::epsilon();
}

template <typename real> inline
real length( real x, real y, real z ) noexcept
{
    return std::sqrt(x*x+y*y+z*z);

    // hypot is much slower than sqrt; we simply assume non-pathological cases.
    //return std::hypot(x,y,z);
}

// Some linear algebra routines. These work on matrices that are
// so small that it does not make sense to call LAPACK.
// Instead we pass the matrices's size as template arguments and
// let the compiler do its magic.
template <typename real> inline
real generate_givens_rotation( real a, real b ) noexcept
{
    using std::abs;
    using std::sqrt;

    real c, s, rho, tau;
    if ( b == 0 )
    {
        c = 1; s = 0;
    }
    else
    {
        if ( abs(b) > abs(a) ) { tau = -a/b; s = 1/sqrt(1+tau*tau); c = s*tau; }
        else                   { tau = -b/a; c = 1/sqrt(1+tau*tau); s = c*tau; }
    }

    if ( c == 0 )
    {
        rho = 1;
    }
    else if ( abs(s) < abs(c) )
    {
        rho = (c>0) ? s/2 : -s/2;
    }
    else
    {
        rho = (s>0) ? 2/c : -2/c;
    }
    return rho;
}

template <typename real> inline
void givens_rotation_from_rho( real rho, real &c, real &s ) noexcept
{
    using std::abs;
    using std::sqrt;

    if ( rho == 1 )
    {
        c = 0; s = 1;
    }
    else if ( abs(rho) < 1 )
    {
        s = 2*rho; c = sqrt(1-s*s);
    }
    else
    {
        c = 2/rho; s = sqrt(1-c*c);
    }
}

template <typename real, size_t m, size_t n>
void givens_qr( real *A ) noexcept
{
    // Computes the QR-decomposition of the given matrix A of
    // m rows and n colums in place, using Givens rotations.
    real rho, c, s, t1, t2;

    for ( size_t j = 0; j < n; ++j )
    {
        for ( size_t i = m - 1; i > j; --i )
        {
            rho = generate_givens_rotation( A[m*j + i-1], A[m*j + i] );
            givens_rotation_from_rho(rho,c,s);

            for ( size_t k = j; k < n; ++k )
            {
                t1 = A[ m*k + i-1 ]; t2 = A[ m*k + i ];
                A[ m*k + i-1 ] = c*t1 - s*t2;
                A[ m*k + i   ] = s*t1 + c*t2;
            }
            A[ m*j + i ] = rho;
        }
    }
}

template <typename real, size_t m, size_t n>
real estimate_condition( const real *QR ) noexcept
{
    // Given the QR decomposition of a matrix A, this method
    // computes an estimate of the condition number of A.
    // Golub, van Loan, Matrix Computations, 4th edition,
    // Algorithm 3.5.1

    using std::max;
    using std::abs;

    // Check for actual zeros.
    for ( size_t k = 0; k < n; ++k )
    {
        if ( QR[m*k+k] == 0 )
            return 2/eps<real>();
    }

    real p[n] {};
    real y[n] {};
    for ( size_t k = n; k-- > 0; )
    {
        real y_plus  = ( 1-p[k])/QR[m*k + k];
        real y_minus = (-1-p[k])/QR[m*k + k];
        real p_plus  = 0, p_minus = 0;
        for ( size_t j = 0; j < k; ++j )
        {
            p_plus  += abs( p[j] + QR[m*k+j]*y_plus  );
            p_minus += abs( p[j] + QR[m*k+j]*y_minus );
        }

        if ( abs(y_plus) + p_plus > abs(y_minus) + p_minus)
        {
            y[k] = y_plus;
            for ( size_t j = 0; j < k; ++j )
                p[j] = p[j] + QR[m*k+j]*y_plus;
        }
        else
        {
            y[k] = y_minus;
            for ( size_t j = 0; j < k; ++j )
                p[j] = p[j] + QR[m*k+j]*y_minus;
        }
    }

    real norm_y = 0, norm_R = 0;
    for ( size_t i = 0; i < n; ++i )
    {
        norm_y = max(norm_y,y[i]);
        real row_sum = 0;
        for ( size_t j = i; j < n; ++j )
            row_sum += abs(QR[i+j*m]);
        norm_R = max(norm_R,row_sum);
    }
    return norm_y*norm_R;
}

template <typename real, size_t m, size_t n>
void least_squares_solve( const real *A, real *b ) noexcept
{
    // Given QR-decomposition of A, dimensions (m,n) with m >= n.
    // This method computes the least squares solution of Ax = b.

    // Compute y  = Q^T b.
    for ( size_t j = 0; j < n; ++j )
    {
        for ( size_t i = m - 1; i > j; --i )
        {
           real rho = A[ m*j + i ];
           real c, s; givens_rotation_from_rho(rho,c,s);

           real t1 = b[i-1], t2 = b[i];
           b[ i-1 ] = c*t1 - s*t2;
           b[ i   ] = s*t1 + c*t2;
        }
    }

    // Solve Rx = y.
    for ( size_t i = n; i-- > 0; )
    {
        for ( size_t j = i + 1; j < n; ++j )
        {
            b[i] -= A[ m*j + i ]*b[j];
        }
        b[i] /= A[ m*i + i ];
    }
}

template <typename real, size_t m, size_t n>
void transposed_least_squares_solve( const real *A, real *a, real *b ) noexcept
{
    // Given QR-decomposition of A, dimensions (m,n) with m >= n.
    // This method computes the minimum norm solution of A^T x = a, A^T x = b.
    // Golub van Loan, Algorithm 5.6.2.

    // L := R^T;
    // Solve Ly = a and Ly = b respectively.
    for ( size_t i = 0; i < n; ++i )
    {
        for ( size_t j = 0; j < i; ++j )
        {
            a[i] -= A[m*i + j]*a[j];
            b[i] -= A[m*i + j]*b[j];
        }
        a[i] /= A[m*i + i];
        b[i] /= A[m*i + i];
    }
    for ( size_t i = n; i < m; ++i )
        a[i] = b[i] = 0;

    // Compute x = Qy.
    for ( size_t j = n; j-- > 0; )
    {
        for ( size_t i = j + 1; i < m; ++i )
        {
           real rho = A[ m*j + i ];
           real c, s; givens_rotation_from_rho(rho,c,s);

           real t1 = a[i-1], t2 = a[i];
           a[ i-1 ] =  c*t1 + s*t2;
           a[ i   ] = -s*t1 + c*t2;

           t1 = b[i-1]; t2 = b[i];
           b[ i-1 ] =  c*t1 + s*t2;
           b[ i   ] = -s*t1 + c*t2;
        }
    }
}

template <typename real>
class basis
{
public:
    basis() = default;
    basis( sphere<real> S );
    basis( sphere<real> S0, sphere<real> S1 );
    basis( sphere<real> S0, sphere<real> S1, sphere<real> S2 );
    basis( sphere<real> S0, sphere<real> S1, sphere<real> S2, sphere<real> S3 );

    sphere<real> operator()( size_t i ) noexcept { return V[i]; }

    sphere<real> ball()   { update(); return bounds; }

    size_t size() const noexcept { return n;    }
    bool is_violated_by( sphere<real> S ) noexcept;
    bool is_violated_by( const basis<real> &B ) noexcept;

private:
    size_t       n { 0 };
    sphere<real> V[ 4 ];

    void   update();
    bool   clean { false };
    sphere<real> bounds;
};

template <typename real>
basis<real>::basis( sphere<real> S )
{
    V[0] = S; n = 1;
    bounds = S;
    clean = true;
}

template <typename real>
basis<real>::basis( sphere<real> S0, sphere<real> S1 )
{
    V[0] = S0;
    V[1] = S1;
    n = 2;
    clean = false;
}

template <typename real>
basis<real>::basis( sphere<real> S0, sphere<real> S1,
                    sphere<real> S2 )
{
    V[0] = S0;
    V[1] = S1;
    V[2] = S2;
    n = 3;
    clean = false;
}

template <typename real>
basis<real>::basis( sphere<real> S0, sphere<real> S1,
                    sphere<real> S2, sphere<real> S3 )
{
    V[0] = S0;
    V[1] = S1;
    V[2] = S2;
    V[3] = S3;
    n = 4;
    clean = false;
}

template <typename real>
void basis<real>::update()
{
    using std::max;
    using std::abs;
    using std::sqrt;

    while ( clean == false )
    {
        if ( n == 1 )
        {
            bounds = V[0];
            clean  = true;
            return;
        }

        const real r0 = V[0].r;
        real max_rho = r0;
        real AQR[3*3], a[3], b[3];
        for ( size_t i = 1; i < n; ++i )
        {
            AQR[ 0 + 3*(i-1) ] = (V[i].x - V[0].x);
            AQR[ 1 + 3*(i-1) ] = (V[i].y - V[0].y);
            AQR[ 2 + 3*(i-1) ] = (V[i].z - V[0].z);

            real ri = V[i].r;
            a[i-1] = ri - r0;
            b[i-1] = ( (V[i].x-V[0].x)*(V[i].x-V[0].x) +
                       (V[i].y-V[0].y)*(V[i].y-V[0].y) + 
                       (V[i].z-V[0].z)*(V[i].z-V[0].z) + r0*r0 - ri*ri)/2;
            max_rho = max( max_rho, ri );
        }

        real condition = 0;
        switch (n)
        {
        case 2: givens_qr<real,3,1>(AQR);
                condition = estimate_condition<real,3,1>(AQR);
                transposed_least_squares_solve<real,3,1>(AQR,a,b);
                break;
        case 3: givens_qr<real,3,2>(AQR);
                condition = estimate_condition<real,3,2>(AQR);
                transposed_least_squares_solve<real,3,2>(AQR,a,b);
                break;
        case 4: givens_qr<real,3,3>(AQR);
                condition = estimate_condition<real,3,3>(AQR);
                transposed_least_squares_solve<real,3,3>(AQR,a,b);
                break;
        }
        if ( condition*eps<real>() > 1 )
        {
            n--; continue;
        }

        // At this point we know the centre of the desired sphere is
        // V[0].centre + b + rho*a. It remains to find rho. Rho is the
        // solution of a quadratic equation, there are thus two possible
        // solutions: rho_minus and rho_plus.
        const real p = (r0 + a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) /
                       (1  - a[0]*a[0] - a[1]*a[1] - a[2]*a[2]);
        const real q = (b[0]*b[0] + b[1]*b[1] + b[2]*b[2] - r0*r0)/
                       (1  - a[0]*a[0] - a[1]*a[1] - a[2]*a[2]);
        const real root = sqrt(abs(p*p+q));
        const real rho_minus = p - root;
        const real rho_plus  = p + root;

        const real z_minus[3] = { rho_minus*a[0]+b[0], rho_minus*a[1]+b[1], rho_minus*a[2]+b[2] };
        const real z_plus [3] = { rho_plus *a[0]+b[0], rho_plus *a[1]+b[1], rho_plus *a[2]+b[2] };

        auto in_convex_hull = [this,AQR]( const real x[3] ) -> bool
        {
            // Solve Al = z.
            real l[4] = { real(1), x[0], x[1], x[2] };
            switch ( n )
            {
            case 2: least_squares_solve<real,3,1>(AQR,l+1); break;
            case 3: least_squares_solve<real,3,2>(AQR,l+1); break;
            case 4: least_squares_solve<real,3,3>(AQR,l+1); break;
            }

            for ( size_t i = 1; i < n; ++i )
                l[0] -= l[i];

            for ( size_t i = 0; i < n; ++i )
            {
                if ( l[i] < -eps<real>() || l[i] > 1+eps<real>() )
                    return false;
            }
            return true;
        };

        // Try if one of the solutions is feasible: it must be in the convex 
        // hull of the basis members. Otherwise shrink and try again.
        if ( rho_minus >= max_rho && in_convex_hull(z_minus) )
        {
            bounds.r = rho_minus;
            bounds.x = V[0].x + z_minus[0];
            bounds.y = V[0].y + z_minus[1];
            bounds.z = V[0].z + z_minus[2];
            clean = true;
        }
        else if ( rho_plus >= max_rho && in_convex_hull(z_plus) )
        {
            bounds.r = rho_plus;
            bounds.x = V[0].x + z_plus[0];
            bounds.y = V[0].y + z_plus[1];
            bounds.z = V[0].z + z_plus[2];
            clean = true;
        }
        else
        {
            n--; continue;
        }
    }
}

template <typename real> inline
bool basis<real>::is_violated_by( sphere<real> S ) noexcept
{
    using std::max;

    for ( size_t i = 0; i < n; ++i )
    {
        if ( V[i].x == S.x &&
             V[i].y == S.y &&
             V[i].z == S.z &&
             V[i].r == S.r    )
            return false;
    }

    update();
    real dx = S.x-bounds.x;
    real dy = S.y-bounds.y;
    real dz = S.z-bounds.z;
    real excess = max( real(0), length(dx,dy,dz) + S.r - bounds.r );
    return ( excess*excess > eps<real>()*bounds.r*bounds.r );
}

template <typename real> inline
bool basis<real>::is_violated_by( const basis<real> &B ) noexcept
{
    for ( size_t i = 0; i < B.size(); ++i )
    {
        if ( is_violated_by(B.V[i]) )
            return true;
    }
    return false;
}

template <typename real> 
basis<real> basis_computation( basis<real> V, sphere<real> B )
{
    basis<real> candidates[15]; size_t num { 0 };

    switch ( V.size() )
    {
    case 1:
        candidates[0] = basis<real> { B };
        candidates[1] = basis<real> { B, V(0) };
        num = 2;
        break;
    case 2:
        candidates[0] = basis<real> { B };
        candidates[1] = basis<real> { B, V(0) };
        candidates[2] = basis<real> { B, V(1) };
        candidates[3] = basis<real> { B, V(1), V(0) };
        num = 4;
        break;
    case 3:
        candidates[0] = basis<real> { B };
        candidates[1] = basis<real> { B, V(0) };
        candidates[2] = basis<real> { B, V(1) };
        candidates[3] = basis<real> { B, V(2) };
        candidates[4] = basis<real> { B, V(0), V(1) };
        candidates[5] = basis<real> { B, V(0), V(2) };
        candidates[6] = basis<real> { B, V(1), V(2) };
        candidates[7] = basis<real> { B, V(0), V(1), V(2) };
        candidates[8] = basis<real> { B, V(1), V(0), V(2) };
        candidates[9] = basis<real> { B, V(2), V(0), V(1) };
        num = 10;
        break;
    case 4:
        candidates[ 0] = basis<real> { B };
        candidates[ 1] = basis<real> { B, V(0) };
        candidates[ 2] = basis<real> { B, V(1) };
        candidates[ 3] = basis<real> { B, V(2) };
        candidates[ 4] = basis<real> { B, V(3) };
        candidates[ 5] = basis<real> { B, V(0), V(1) };
        candidates[ 6] = basis<real> { B, V(0), V(2) };
        candidates[ 7] = basis<real> { B, V(0), V(3) };
        candidates[ 8] = basis<real> { B, V(1), V(2) };
        candidates[ 9] = basis<real> { B, V(1), V(3) };
        candidates[10] = basis<real> { B, V(2), V(3) };
        candidates[11] = basis<real> { B, V(1), V(2), V(3) };
        candidates[12] = basis<real> { B, V(0), V(2), V(3) };
        candidates[13] = basis<real> { B, V(0), V(1), V(3) };
        candidates[14] = basis<real> { B, V(0), V(1), V(2) };
        num = 15;
        break;
    }

    for ( size_t i = 0; i < num; ++i )
    {
        if ( ! candidates[i].is_violated_by(V) )
        {
            return candidates[i];
        }
    }

    throw std::runtime_error { "solidfmm::bounding_sphere: "
                               "Could not compute a new basis." };
}


// Computes the minimal bounding sphere of a set of points.
// See Kaspar Fischer and Bernd Gärtner:
// "The Smallest Enclosing Ball of Balls: Combinatorial Structure and Algorithms".
template <typename real>
sphere<real> bounding_sphere_impl( size_t n, const real *x, const real *y, const real *z )
{
    if ( n == 0 )
        throw std::logic_error { "solidfmm::bounding_sphere: "
              "Cannot compute bounding sphere of an empty set." };

    basis<real> V { sphere<real> { x[0], y[0], z[0], 0 } };
    while ( true )
    {
        sphere<real> B = V.ball(); // Current candidate of bounding sphere.
        real dx = x[0] - B.x;
        real dy = y[0] - B.y;
        real dz = z[0] - B.z;
        real max_dist = length(dx,dy,dz);

        size_t farthest = 0;
        for ( size_t i = 1; i < n; ++i )
        {
            dx = x[i] - B.x;
            dy = y[i] - B.y;
            dz = z[i] - B.z;
            real dist = length(dx,dy,dz);
            if ( dist > max_dist )
            {
                max_dist = dist;
                farthest = i;
            }
        }

        sphere<real> farthest_sphere { x[farthest], y[farthest], z[farthest], 0 };

        if ( V.is_violated_by( farthest_sphere ) )
        {
            V = basis_computation( V, farthest_sphere );
        }
        else return B;
    }
}


// Computes the minimal bounding sphere of a set of spheres.
// See Kaspar Fischer and Bernd Gärtner:
// "The Smallest Enclosing Ball of Balls: Combinatorial Structure and Algorithms".
template <typename real>
sphere<real> bounding_sphere_impl( const sphere<real> *begin, const sphere<real> *end )
{
    if ( begin == end )
        throw std::logic_error { "solidfmm::bounding_sphere: "
              "Cannot compute bounding sphere of an empty set." };

    basis<real> V { *begin };
    while ( true )
    {
        sphere<real> B = V.ball(); // Current candidate of bounding sphere.
        real dx = begin->x - B.x;
        real dy = begin->y - B.y;
        real dz = begin->z - B.z;
        real max_dist = length(dx,dy,dz) + begin->r;
        const sphere<real>* farthest = begin;
        for ( const sphere<real> *i = begin + 1; i != end; ++i )
        {
            dx = i->x - B.x;
            dy = i->y - B.y;
            dz = i->z - B.z;
            real dist = length(dx,dy,dz) + i->r;
            if ( dist > max_dist )
            {
                max_dist = dist;
                farthest = i;
            }
        }

        if ( V.is_violated_by(*farthest) )
        {
            V = basis_computation(V,*farthest);
        }
        else return B;
    }
}

} // End of anonymous namespace.

sphere<float > bounding_sphere( const sphere<float > *begin, const sphere<float > *end )
{
    return bounding_sphere_impl(begin,end);
}

sphere<double> bounding_sphere( const sphere<double> *begin, const sphere<double> *end )
{
    return bounding_sphere_impl(begin,end);
}

sphere<float > bounding_sphere( size_t n, const float *x, const float *y, const float *z )
{
    return bounding_sphere_impl(n,x,y,z);
}

sphere<double> bounding_sphere( size_t n, const double *x, const double *y, const double *z )
{
    return bounding_sphere_impl(n,x,y,z);
}

} // End of namespace solidfmm.

