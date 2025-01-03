#pragma once

#include <array>
#include <type_traits>
#include <immintrin.h>  // SSE, AVX intrinsics
#ifdef __ARM_NEON
#include <arm_neon.h>    // NEON intrinsics (on ARM)
#endif

namespace simdlib {

template <typename T, size_t N>
struct simd_vector;

template <typename T>
struct is_supported_type : std::false_type {};

template <>
struct is_supported_type<float> : std::true_type {};

template <size_t N>
struct is_power_of_two : std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)> {};

// SSE (4 floats)
template <>
struct simd_vector<float, 4> {
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<4>::value, "size must be a power of 2");

    __m128 data;  // SSE register

    simd_vector() : data(_mm_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm_set1_ps(value)) {}
    explicit simd_vector(__m128 vec) : data(vec) {}

    float operator[](size_t i) const {
        alignas(16) std::array<float, 4> elements;
        _mm_store_ps(elements.data(), data);
        return elements[i];
    }

    simd_vector& operator+=(const simd_vector& other) {
        data = _mm_add_ps(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector& other) const {
        return simd_vector(_mm_add_ps(data, other.data));
    }

    simd_vector& operator-=(const simd_vector& other) {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector& other) const {
        return simd_vector(_mm_sub_ps(data, other.data));
    }

    simd_vector& operator*=(const simd_vector& other) {
        data = _mm_mul_ps(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector& other) const {
        return simd_vector(_mm_mul_ps(data, other.data));
    }

    simd_vector& operator/=(const simd_vector& other) {
        data = _mm_div_ps(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector& other) const {
        return simd_vector(_mm_div_ps(data, other.data));
    }

    simd_vector operator==(const simd_vector& other) const { 
        return simd_vector(_mm_cmpeq_ps(data, other.data));
    }

    simd_vector operator!=(const simd_vector& other) const { 
        return simd_vector(_mm_cmpneq_ps(data, other.data));
    }

    simd_vector operator<(const simd_vector& other) const { 
        return simd_vector(_mm_cmplt_ps(data, other.data));
    }

    simd_vector operator<=(const simd_vector& other) const { 
        return simd_vector(_mm_cmple_ps(data, other.data));
    }

    simd_vector operator>(const simd_vector& other) const { 
        return simd_vector(_mm_cmpgt_ps(data, other.data));
    }

    simd_vector operator>=(const simd_vector& other) const { 
        return simd_vector(_mm_cmpge_ps(data, other.data));
    }
};

// AVX (8 floats)
template <>
struct simd_vector<float, 8> {
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<8>::value, "size must be a power of 2");

    __m256 data;  // AVX register

    simd_vector() : data(_mm256_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm256_set1_ps(value)) {}
    explicit simd_vector(__m256 vec) : data(vec) {}

    float operator[](size_t i) const {
        alignas(32) std::array<float, 8> elements;
        _mm256_store_ps(elements.data(), data);
        return elements[i];
    }

    simd_vector& operator+=(const simd_vector& other) {
        data = _mm256_add_ps(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector& other) const {
        return simd_vector(_mm256_add_ps(data, other.data));
    }
    
    simd_vector& operator-=(const simd_vector& other) {
        data = _mm256_sub_ps(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector& other) const {
        return simd_vector(_mm256_sub_ps(data, other.data));
    }

    simd_vector& operator*=(const simd_vector& other) {
        data = _mm256_mul_ps(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector& other) const {
        return simd_vector(_mm256_mul_ps(data, other.data));
    }

    simd_vector& operator/=(const simd_vector& other) {
        data = _mm256_div_ps(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector& other) const {
        return simd_vector(_mm256_div_ps(data, other.data));
    }

    simd_vector operator==(const simd_vector& other) const { 
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_EQ_OQ));
    }

    simd_vector operator!=(const simd_vector& other) const { 
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_NEQ_OQ));
    }

    simd_vector operator<(const simd_vector& other) const {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_LT_OQ));
    }

    simd_vector operator<=(const simd_vector& other) const { 
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_LE_OQ));
    }

    simd_vector operator>(const simd_vector& other) const { 
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_GT_OQ));
    }

    simd_vector operator>=(const simd_vector& other) const { 
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_GE_OQ));
    }
};

// NEON (4 floats, for ARM)
#ifdef __ARM_NEON
template <>
struct simd_vector<float, 4> {
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<4>::value, "size must be a power of 2");

    float32x4_t data;  // NEON register

    simd_vector() : data(vdupq_n_f32(0.0f)) {}
    explicit simd_vector(float value) : data(vdupq_n_f32(value)) {}
    explicit simd_vector(float32x4_t vec) : data(vec) {}

    float operator[](size_t i) const {
        std::array<float, 4> elements;
        vst1q_f32(elements.data(), data);
        return elements[i];
    }

    simd_vector& operator+=(const simd_vector& other) {
        data = vaddq_f32(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector& other) const {
        return simd_vector(vaddq_f32(data, other.data));
    }

    simd_vector& operator-=(const simd_vector& other) {
        data = vsubq_f32(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector& other) const {
        return simd_vector(vsubq_f32(data, other.data));
    }

    simd_vector& operator*=(const simd_vector& other) {
        data = vmulq_f32(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector& other) const {
        return simd_vector(vmulq_f32(data, other.data));
    }

    simd_vector& operator/=(const simd_vector& other) {
        data = vdivq_f32(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector& other) const {
        return simd_vector(vdivq_f32(data, other.data));
    }

    simd_vector operator==(const simd_vector& other) const { // Added
        return simd_vector(vceqq_f32(data, other.data));
    }

    simd_vector operator!=(const simd_vector& other) const { 
        return simd_vector(vmvnq_u32(vceqq_f32(data, other.data)));
    }

    simd_vector operator<(const simd_vector& other) const { 
        return simd_vector(vcltq_f32(data, other.data));
    }

    simd_vector operator<=(const simd_vector& other) const { 
        return simd_vector(vcleq_f32(data, other.data));
    }

    simd_vector operator>(const simd_vector& other) const { 
        return simd_vector(vcgtq_f32(data, other.data));
    }

    simd_vector operator>=(const simd_vector& other) const { 
        return simd_vector(vcgeq_f32(data, other.data));
    }
};
#endif

// factory function to create a SIMD vector from a scalar value
template <typename T, size_t N>
constexpr simd_vector<T, N> make_vector(T value) {
    static_assert(is_supported_type<T>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<N>::value, "size must be a power of 2");
    return simd_vector<T, N>(value);
}

// SSE
template <>
inline simd_vector<float, 4> make_vector<float, 4>(float value) {
    return simd_vector<float, 4>(value);
}

// AVX
template <>
inline simd_vector<float, 8> make_vector<float, 8>(float value) {
    return simd_vector<float, 8>(value);
}

// NEON
#ifdef __ARM_NEON
template <>
inline simd_vector<float, 4> make_vector<float, 4>(float value) {
    return simd_vector<float, 4>(value);
}
#endif

} // namespace simdlib