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
#include <iomanip>
#include <iostream>
#include <solidfmm/swap_matrix.hpp>
#include <solidfmm/operator_data.hpp>

// Values taken from Dehnen, equation (64).
const double B4[9][9] =
{
    {   1,  -8,  28, -56,  70, -56,  28,  -8,   1 },
    {  -1,   6, -14,  14,   0, -14,  14,  -6,   1 },
    {   1,  -4,   4,   4, -10,   4,   4,  -4,   1 },
    {  -1,   2,   2,  -6,   0,   6,  -2,  -2,   1 },
    {   1,   0,  -4,   0,   6,   0,  -4,   0,   1 },
    {  -1,  -2,   2,   6,   0,  -6,  -2,   2,   1 },
    {   1,   4,   4,  -4, -10,  -4,   4,   4,   1 },
    {  -1,  -6, -14, -14,   0,  14,  14,   6,   1 },
    {   1,   8,  28,  56,  70,  56,  28,   8,   1 }
};

int main()
{
    solidfmm::operator_data<double> data(16);
    solidfmm::swap_matrix<double> B(12);

    std::cout << std::setprecision(4) << std::scientific;

    std::cout << "Absolute errors in Matrix B(4,i,j):\n";
    double max_error = 0;
    for ( int i = -4; i <= 4; ++i )
    {
        for ( int j = -4; j <= 4; ++j )
        {
            double b_ref = B4[4+i][4+j]/double(16);
            double b_lib = B(4,i,j);
            double err   = std::abs(b_ref-b_lib);
            max_error    = std::max(max_error,err);
            std::cout << std::setw(12) << err << ' ';
        }
        std::cout << "\n";
    } 
    std::cout << "Maximum error: " << max_error << std::endl;
    return max_error != 0;
}

