/*
 * Copyright (C) 2021, 2022 Matthias Kirchhart
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

#include <cmath>

namespace solidfmm
{

namespace
{

template <typename real>
solid<real> R_impl( size_t P, real x, real y, real z )
{
    if ( P == 0 ) return solid<real> {};

    solid<real> result(P);
    result.re(0,0) = 1;

    if ( P == 1 ) return result;

    result.re(1,0) = z;
    result.re(1,1) = x/2;
    result.im(1,1) = y/2;

    // First do the diagonal.
    for ( size_t m = 2; m < P; ++m )
    {
        real a = result.re(m-1,m-1);
        real b = result.im(m-1,m-1);
        real fac = real(1)/real(2*m);

        result.re(m,m) = (a*x - b*y)*fac;
        result.im(m,m) = (a*y + b*x)*fac;
    }

    // Afterwards the "small diagonal" below.
    for ( size_t m = 1; m < P - 1; ++m )
    {
        result.re(m+1,m) = z*result.re(m,m);
        result.im(m+1,m) = z*result.im(m,m);
    }

    // Now the rest.
    const real r2 { x*x + y*y + z*z };
    for ( size_t m = 0; m < P - 2; ++m )
    for ( size_t n = m + 2; n < P; ++n )
    {
        real a = result.re(n-1,m);
        real b = result.re(n-2,m);
        real c = result.im(n-1,m);
        real d = result.im(n-2,m);
        real fac = real(1)/( (real) (n+m)*(n-m));

        result.re(n,m) = ( a*z*(2*n-1) - r2*b ) * fac;
        result.im(n,m) = ( c*z*(2*n-1) - r2*d ) * fac;
    }

    return result;
}

template <typename real>
solid<real> dR_impl( size_t P, real x, real y, real z )
{
    // This is just forward automatic differentiation,
    // explicitly written down.
    if ( P < 2 ) return solid<real> { P, 3 };

    solid<real> val(P), grad(P,3);
    val.re(0,0) = 1;

    val.re(1,0) = z;   grad.re(1,0,2) = 1;
    val.re(1,1) = x/2; grad.re(1,1,0) = 0.5;
    val.im(1,1) = y/2; grad.im(1,1,1) = 0.5;

    // First do the diagonal.
    for ( size_t m = 2; m < P; ++m )
    {
        real a  =  val.re(m-1,m-1);
        real ax = grad.re(m-1,m-1,0);
        real ay = grad.re(m-1,m-1,1);
        real az = grad.re(m-1,m-1,2);

        real b  =  val.im(m-1,m-1);
        real bx = grad.im(m-1,m-1,0);
        real by = grad.im(m-1,m-1,1);
        real bz = grad.im(m-1,m-1,2);

        real fac = real(1)/real(2*m);

        val.re(m,m)    = (a*x - b*y)*fac;
        grad.re(m,m,0) = ( ax*x + a - bx*y ) * fac;
        grad.re(m,m,1) = ( ay*x - by*y - b ) * fac;
        grad.re(m,m,2) = ( az*x - bz*y ) * fac;

         val.im(m,m)   = ( a*y + b*x)*fac;
        grad.im(m,m,0) = (ax*y + bx*x + b ) * fac;
        grad.im(m,m,1) = (ay*y + a + by*x ) * fac;
        grad.im(m,m,2) = (az*y + bz*x) * fac;
    }

    // Afterwards the "small diagonal" below.
    for ( size_t m = 1; m < P - 1; ++m )
    {
         val.re(m+1,m)   = z*val.re(m,m);
        grad.re(m+1,m,0) = z*grad.re(m,m,0);
        grad.re(m+1,m,1) = z*grad.re(m,m,1);
        grad.re(m+1,m,2) = val.re(m,m) + z*grad.re(m,m,2);

         val.im(m+1,m)   = z* val.im(m,m);
        grad.im(m+1,m,0) = z*grad.im(m,m,0);
        grad.im(m+1,m,1) = z*grad.im(m,m,1);
        grad.im(m+1,m,2) = val.im(m,m) + z*grad.im(m,m,2);
    }

    // Now the rest.
    const real r2  { x*x + y*y + z*z };
    const real r2x { 2*x }, r2y { 2*y }, r2z { 2*z };
    for ( size_t m = 0; m < P - 2; ++m )
    for ( size_t n = m + 2; n < P; ++n )
    {
        real a  =  val.re(n-1,m);
        real ax = grad.re(n-1,m,0);
        real ay = grad.re(n-1,m,1);
        real az = grad.re(n-1,m,2);

        real b  =  val.re(n-2,m);
        real bx = grad.re(n-2,m,0);
        real by = grad.re(n-2,m,1);
        real bz = grad.re(n-2,m,2);

        real c  =  val.im(n-1,m);
        real cx = grad.im(n-1,m,0);
        real cy = grad.im(n-1,m,1);
        real cz = grad.im(n-1,m,2);

        real d  =  val.im(n-2,m);
        real dx = grad.im(n-2,m,0);
        real dy = grad.im(n-2,m,1);
        real dz = grad.im(n-2,m,2);
        real fac = real(1)/( (real) (n+m)*(n-m));

         val.re(n,m)   = ( a*z*(2*n-1) - r2*b ) * fac;
        grad.re(n,m,0) = (ax*z*(2*n-1) - r2x*b - r2*bx ) * fac;
        grad.re(n,m,1) = (ay*z*(2*n-1) - r2y*b - r2*by ) * fac;
        grad.re(n,m,2) = (az*z*(2*n-1) + a*(2*n-1) - r2z*b - r2*bz ) * fac;

         val.im(n,m)   = (  c*z*(2*n-1) - r2 *d ) * fac;
        grad.im(n,m,0) = ( cx*z*(2*n-1) - r2x*d - r2*dx ) * fac;
        grad.im(n,m,1) = ( cy*z*(2*n-1) - r2y*d - r2*dy ) * fac;
        grad.im(n,m,2) = ( cz*z*(2*n-1) + c*(2*n-1) - r2z*d - r2*dz ) * fac;
    }

    return grad;
}

template <typename real>
solid<real> S_impl( size_t P, real x, real y, real z )
{
    if ( P == 0 ) return solid<real> {};

    solid<real> result(P);
    real r2inv = 1/(x*x + y*y + z*z);
    result.re(0,0) = std::sqrt(r2inv);

    if ( P == 1 ) return result;

    result.re(1,0) = z*r2inv*result.re(0,0);
    result.re(1,1) = x*r2inv*result.re(0,0);
    result.im(1,1) = y*r2inv*result.re(0,0);

    // First do the diagonal.
    for ( size_t m = 2; m < P; ++m )
    {
        real a = result.re(m-1,m-1);
        real b = result.im(m-1,m-1);
        real fac = real(2*m-1)*r2inv;

        result.re(m,m) = (a*x - b*y)*fac;
        result.im(m,m) = (a*y + b*x)*fac;
    }

    // Afterwards the "small diagonal" below.
    for ( size_t m = 1; m < P - 1; ++m )
    {
        real fac = (2*m+1)*z*r2inv;
        result.re(m+1,m) = fac*result.re(m,m);
        result.im(m+1,m) = fac*result.im(m,m);
    }

    // Now the rest.
    for ( size_t m = 0; m < P - 2; ++m )
    for ( size_t n = m + 2; n < P; ++n )
    {
        real a = result.re(n-1,m);
        real b = result.re(n-2,m);
        real c = result.im(n-1,m);
        real d = result.im(n-2,m);

        result.re(n,m) = ( a*z*(2*n-1) - (n+m-1)*(n-m-1)*b ) * r2inv;
        result.im(n,m) = ( c*z*(2*n-1) - (n+m-1)*(n-m-1)*d ) * r2inv;
    }

    return result;
}

template <typename real>
solid<real> dS_impl( size_t P, real x, real y, real z )
{
    if ( P == 0 ) return solid<real> { P, 3 };

    solid<real> val { P }, grad { P, 3 };
    const real r2inv  = 1/(x*x + y*y + z*z);
    const real r2invx = -2*x * r2inv * r2inv;
    const real r2invy = -2*y * r2inv * r2inv;
    const real r2invz = -2*z * r2inv * r2inv;
     val.re(0,0)   = std::sqrt(r2inv);
    grad.re(0,0,0) = -x*val.re(0,0)*val.re(0,0)*val.re(0,0);
    grad.re(0,0,1) = -y*val.re(0,0)*val.re(0,0)*val.re(0,0);
    grad.re(0,0,2) = -z*val.re(0,0)*val.re(0,0)*val.re(0,0);

    if ( P == 1 ) return grad;

     val.re(1,0)   = z*r2inv *val.re(0,0);
    grad.re(1,0,0) = z*r2invx*val.re(0,0) + z*r2inv*grad.re(0,0,0);
    grad.re(1,0,1) = z*r2invy*val.re(0,0) + z*r2inv*grad.re(0,0,1);
    grad.re(1,0,2) = z*r2invz*val.re(0,0) + z*r2inv*grad.re(0,0,2) + r2inv*val.re(0,0);

     val.re(1,1)   = x*r2inv *val.re(0,0);
    grad.re(1,1,0) = x*r2invx*val.re(0,0) + x*r2inv*grad.re(0,0,0) + r2inv*val.re(0,0);
    grad.re(1,1,1) = x*r2invy*val.re(0,0) + x*r2inv*grad.re(0,0,1);
    grad.re(1,1,2) = x*r2invz*val.re(0,0) + x*r2inv*grad.re(0,0,2);
    
     val.im(1,1)   = y*r2inv *val.re(0,0);
    grad.im(1,1,0) = y*r2invx*val.re(0,0) + y*r2inv*grad.re(0,0,0);
    grad.im(1,1,1) = y*r2invy*val.re(0,0) + y*r2inv*grad.re(0,0,1) + r2inv*val.re(0,0);
    grad.im(1,1,2) = y*r2invz*val.re(0,0) + y*r2inv*grad.re(0,0,2);

    // First do the diagonal.
    for ( size_t m = 2; m < P; ++m )
    {
        real a  =  val.re(m-1,m-1);
        real ax = grad.re(m-1,m-1,0);
        real ay = grad.re(m-1,m-1,1);
        real az = grad.re(m-1,m-1,2);

        real b  =  val.im(m-1,m-1);
        real bx = grad.im(m-1,m-1,0);
        real by = grad.im(m-1,m-1,1);
        real bz = grad.im(m-1,m-1,2);

        real fac  = real(2*m-1)*r2inv;
        real facx = real(2*m-1)*r2invx;
        real facy = real(2*m-1)*r2invy;
        real facz = real(2*m-1)*r2invz;

         val.re(m,m)   = (a*x - b*y)*fac;
        grad.re(m,m,0) = (ax*x + a - bx*y)*fac + (a*x - b*y)*facx;
        grad.re(m,m,1) = (ay*x - by*y - b)*fac + (a*x - b*y)*facy;
        grad.re(m,m,2) = (az*x - bz*y)    *fac + (a*x - b*y)*facz;

         val.im(m,m)   = (a*y + b*x)*fac;
        grad.im(m,m,0) = (ax*y + bx*x + b)*fac + (a*y + b*x)*facx;
        grad.im(m,m,1) = (ay*y + a + by*x)*fac + (a*y + b*x)*facy;
        grad.im(m,m,2) = (az*y + bz*x)    *fac + (a*y + b*x)*facz;
    }

    // Afterwards the "small diagonal" below.
    for ( size_t m = 1; m < P - 1; ++m )
    {
        real fac  = (2*m+1)*z*r2inv;
        real facx = (2*m+1)*z*r2invx;
        real facy = (2*m+1)*z*r2invy;
        real facz = (2*m+1)*(z*r2invz  + r2inv);

         val.re(m+1,m)   = fac *val.re(m,m);
        grad.re(m+1,m,0) = facx*val.re(m,m) + fac*grad.re(m,m,0);
        grad.re(m+1,m,1) = facy*val.re(m,m) + fac*grad.re(m,m,1);
        grad.re(m+1,m,2) = facz*val.re(m,m) + fac*grad.re(m,m,2);

         val.im(m+1,m)   = fac *val.im(m,m);
        grad.im(m+1,m,0) = facx*val.im(m,m) + fac*grad.im(m,m,0);
        grad.im(m+1,m,1) = facy*val.im(m,m) + fac*grad.im(m,m,1);
        grad.im(m+1,m,2) = facz*val.im(m,m) + fac*grad.im(m,m,2);
    }

    // Now the rest.
    for ( size_t m = 0; m < P - 2; ++m )
    for ( size_t n = m + 2; n < P; ++n )
    {
        real a  =  val.re(n-1,m);
        real ax = grad.re(n-1,m,0);
        real ay = grad.re(n-1,m,1);
        real az = grad.re(n-1,m,2);

        real b  =  val.re(n-2,m);
        real bx = grad.re(n-2,m,0);
        real by = grad.re(n-2,m,1);
        real bz = grad.re(n-2,m,2);

        real c  =  val.im(n-1,m);
        real cx = grad.im(n-1,m,0);
        real cy = grad.im(n-1,m,1);
        real cz = grad.im(n-1,m,2);

        real d  =  val.im(n-2,m);
        real dx = grad.im(n-2,m,0);
        real dy = grad.im(n-2,m,1);
        real dz = grad.im(n-2,m,2);

         val.re(n,m)   = ( a *z*(2*n-1) - (n+m-1)*(n-m-1)*b ) * r2inv;
        grad.re(n,m,0) = ( ax*z*(2*n-1) - (n+m-1)*(n-m-1)*bx) * r2inv +
                         ( a *z*(2*n-1) - (n+m-1)*(n-m-1)*b ) * r2invx;
        grad.re(n,m,1) = ( ay*z*(2*n-1) - (n+m-1)*(n-m-1)*by) * r2inv +
                         ( a *z*(2*n-1) - (n+m-1)*(n-m-1)*b ) * r2invy;
        grad.re(n,m,2) = ( (az*z + a)*(2*n-1) - (n+m-1)*(n-m-1)*bz) * r2inv +
                         ( a *z*(2*n-1) - (n+m-1)*(n-m-1)*b ) * r2invz;
        
         val.im(n,m)   = ( c *z*(2*n-1) - (n+m-1)*(n-m-1)*d ) * r2inv;
        grad.im(n,m,0) = ( cx*z*(2*n-1) - (n+m-1)*(n-m-1)*dx) * r2inv +
                         ( c *z*(2*n-1) - (n+m-1)*(n-m-1)*d ) * r2invx;
        grad.im(n,m,1) = ( cy*z*(2*n-1) - (n+m-1)*(n-m-1)*dy) * r2inv +
                         ( c *z*(2*n-1) - (n+m-1)*(n-m-1)*d ) * r2invy;
        grad.im(n,m,2) = ( (cz*z + c)*(2*n-1) - (n+m-1)*(n-m-1)*dz) * r2inv +
                         ( c *z*(2*n-1) - (n+m-1)*(n-m-1)*d ) * r2invz;
             
    }

    return grad;
}


}

namespace harmonics
{

template <> solid<float > R<float >( size_t P, float  x, float  y, float  z )
{
    return R_impl<float >(P,x,y,z);
}

template <> solid<float > dR<float >( size_t P, float  x, float  y, float  z )
{
    return dR_impl<float >(P,x,y,z);
}

template <> solid<float > S<float >( size_t P, float  x, float  y, float  z )
{
    return S_impl<float >(P,x,y,z);
}

template <> solid<float > dS<float >( size_t P, float  x, float  y, float  z )
{
    return dS_impl<float >(P,x,y,z);
}


template <> solid<double> S<double>( size_t P, double x, double y, double z )
{
    return S_impl<double>(P,x,y,z);
}

template <> solid<double> dS<double>( size_t P, double x, double y, double z )
{
    return dS_impl<double>(P,x,y,z);
}

template <> solid<double> R<double>( size_t P, double x, double y, double z )
{
    return R_impl<double>(P,x,y,z);
}

template <> solid<double> dR<double>( size_t P, double x, double y, double z )
{
    return dR_impl<double>(P,x,y,z);
}

}

}

