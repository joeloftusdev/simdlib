#pragma once

#include <type_traits>
#include <cstddef>

namespace simdlib
{

constexpr size_t SSE_ALIGNMENT = 16;
constexpr size_t AVX_ALIGNMENT = 32;
constexpr size_t SSE_SIZE = 4;
constexpr size_t AVX_SIZE = 8;

template <typename T, size_t N> struct simd_vector;

template <typename T> struct is_supported_type : std::false_type
{
};

template <> struct is_supported_type<float> : std::true_type
{
};

template <size_t N>
struct is_power_of_two : std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)>
{
};
}