#pragma once

#include "simd_vector.hpp"
#include <iostream>

namespace simdlib
{

// print the vector
template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const simd_vector<T, N> &vec)
{
    os << "[";
    for (size_t i = 0; i < N; ++i)
    {
        os << vec[i];
        if (i < N - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

} // namespace simdlib
