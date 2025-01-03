#pragma once

#include <immintrin.h>  // SSE, AVX intrinsics
#ifdef __ARM_NEON
#include <arm_neon.h>    // NEON intrinsics (on ARM)
#endif

namespace simdlib {

template <typename T, size_t N>
struct simd_vector;

// SSE (4 floats)
template <>
struct simd_vector<float, 4> {
    __m128 data;  // SSE register

    simd_vector() : data(_mm_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm_set1_ps(value)) {}
    explicit simd_vector(__m128 vec) : data(vec) {}

    float operator[](size_t i) const {
        alignas(16) float elements[4];
        _mm_store_ps(elements, data);
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
};

// AVX (8 floats)
template <>
struct simd_vector<float, 8> {
    __m256 data;  // AVX register

    simd_vector() : data(_mm256_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm256_set1_ps(value)) {}
    explicit simd_vector(__m256 vec) : data(vec) {}

    float operator[](size_t i) const {
        alignas(32) float elements[8];
        _mm256_store_ps(elements, data);
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
};

// NEON (4 floats, for ARM)
#ifdef __ARM_NEON
template <>
struct simd_vector<float, 4> {
    float32x4_t data;  // NEON register

    simd_vector() : data(vdupq_n_f32(0.0f)) {}
    explicit simd_vector(float value) : data(vdupq_n_f32(value)) {}
    explicit simd_vector(float32x4_t vec) : data(vec) {}

    float operator[](size_t i) const {
        float elements[4];
        vst1q_f32(elements, data);
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
};
#endif

// factory function to create a SIMD vector from a scalar value
template <typename T, size_t N>
simd_vector<T, N> make_vector(T value) {
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