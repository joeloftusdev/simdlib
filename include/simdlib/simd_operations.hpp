#pragma once

#include "simd_vector.hpp"

namespace simdlib
{

// add two vectors
template <typename T, size_t N>
simd_vector<T, N> operator+(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs + rhs;
}

// addition in place
template <typename T, size_t N>
simd_vector<T, N> &operator+=(simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    lhs += rhs;
    return lhs;
}

// subtract two vectors
template <typename T, size_t N>
simd_vector<T, N> operator-(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs - rhs;
}

// subtraction in place
template <typename T, size_t N>
simd_vector<T, N> &operator-=(simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    lhs -= rhs;
    return lhs;
}

// multiply two vectors
template <typename T, size_t N>
simd_vector<T, N> operator*(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs * rhs;
}

// multiplication in place
template <typename T, size_t N>
simd_vector<T, N> &operator*=(simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    lhs *= rhs;
    return lhs;
}

// divide two vectors
template <typename T, size_t N>
simd_vector<T, N> operator/(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs / rhs;
}

// division in place
template <typename T, size_t N>
simd_vector<T, N> &operator/=(simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    lhs /= rhs;
    return lhs;
}

// element-wise equality
template <typename T, size_t N>
simd_vector<T, N> operator==(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs == rhs;
}

// element-wise inequality
template <typename T, size_t N>
simd_vector<T, N> operator!=(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs != rhs;
}

// element-wise less than
template <typename T, size_t N>
simd_vector<T, N> operator<(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs < rhs;
}

// element-wise less than or equal to
template <typename T, size_t N>
simd_vector<T, N> operator<=(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs <= rhs;
}

// element-wise greater than
template <typename T, size_t N>
simd_vector<T, N> operator>(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs > rhs;
}

// element-wise greater than or equal to
template <typename T, size_t N>
simd_vector<T, N> operator>=(const simd_vector<T, N> &lhs, const simd_vector<T, N> &rhs)
{
    return lhs >= rhs;
}

// horizontal sum
template <typename T, size_t N>
T horizontal_sum(const simd_vector<T, N> &vec)
{
    return vec.horizontal_sum();
}

// horizontal max
template <typename T, size_t N>
T horizontal_max(const simd_vector<T, N> &vec)
{
    return vec.horizontal_max();
}

// horizontal min
template <typename T, size_t N>
T horizontal_min(const simd_vector<T, N> &vec)
{
    return vec.horizontal_min();
}

} // namespace simdlib