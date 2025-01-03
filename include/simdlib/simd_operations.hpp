#pragma once

#include "simd_vector.hpp"

namespace simdlib {

//add two vectors
template <typename T, size_t N>
simd_vector<T, N> operator+(const simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    return lhs + rhs;
}

//addition in place
template <typename T, size_t N>
simd_vector<T, N>& operator+=(simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

//subtract two vectors
template <typename T, size_t N>
simd_vector<T, N> operator-(const simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    return lhs - rhs;
}

//subtraction in place
template <typename T, size_t N>
simd_vector<T, N>& operator-=(simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
}

//multiply two vectors
template <typename T, size_t N>
simd_vector<T, N> operator*(const simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    return lhs * rhs;
}

//multiplication in place
template <typename T, size_t N>
simd_vector<T, N>& operator*=(simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    lhs *= rhs;
    return lhs;
}

//divide two vectors
template <typename T, size_t N>
simd_vector<T, N> operator/(const simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    return lhs / rhs;
}

//division in place
template <typename T, size_t N>
simd_vector<T, N>& operator/=(simd_vector<T, N>& lhs, const simd_vector<T, N>& rhs) {
    lhs /= rhs;
    return lhs;
}

} // namespace simdlib