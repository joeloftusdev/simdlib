#pragma once

#include <array>
#include <type_traits>
#include <immintrin.h> // SSE, AVX intrinsics
#ifdef __ARM_NEON
#include <arm_neon.h> // NEON intrinsics (on ARM)
#endif

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

// SSE (4 floats)
template <> struct simd_vector<float, SSE_SIZE>
{
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<SSE_SIZE>::value, "size must be a power of 2");

    __m128 data; // SSE register

    simd_vector() : data(_mm_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm_set1_ps(value)) {}
    explicit simd_vector(__m128 vec) : data(vec) {}

    float operator[](size_t i) const
    {
        alignas(SSE_ALIGNMENT) std::array<float, SSE_SIZE> elements{};
        _mm_store_ps(elements.data(), data);
        return elements[i];
    }

    simd_vector &operator+=(const simd_vector &other)
    {
        data = _mm_add_ps(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector &other) const
    {
        return simd_vector(_mm_add_ps(data, other.data));
    }

    simd_vector &operator-=(const simd_vector &other)
    {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector &other) const
    {
        return simd_vector(_mm_sub_ps(data, other.data));
    }

    simd_vector &operator*=(const simd_vector &other)
    {
        data = _mm_mul_ps(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector &other) const
    {
        return simd_vector(_mm_mul_ps(data, other.data));
    }

    simd_vector &operator/=(const simd_vector &other)
    {
        data = _mm_div_ps(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector &other) const
    {
        return simd_vector(_mm_div_ps(data, other.data));
    }

    // Element-wise conditional operations
    simd_vector operator==(const simd_vector &other) const
    {
        return simd_vector(_mm_cmpeq_ps(data, other.data));
    }

    simd_vector operator!=(const simd_vector &other) const
    {
        return simd_vector(_mm_cmpneq_ps(data, other.data));
    }

    simd_vector operator<(const simd_vector &other) const
    {
        return simd_vector(_mm_cmplt_ps(data, other.data));
    }

    simd_vector operator<=(const simd_vector &other) const
    {
        return simd_vector(_mm_cmple_ps(data, other.data));
    }

    simd_vector operator>(const simd_vector &other) const
    {
        return simd_vector(_mm_cmpgt_ps(data, other.data));
    }

    simd_vector operator>=(const simd_vector &other) const
    {
        return simd_vector(_mm_cmpge_ps(data, other.data));
    }

    // Transpose method for 4x4 matrix
    static void transpose(simd_vector &row0, simd_vector &row1, simd_vector &row2,
                          simd_vector &row3)
    {
        __m128 tmp0 = _mm_unpacklo_ps(row0.data, row1.data);
        __m128 tmp1 = _mm_unpackhi_ps(row0.data, row1.data);
        __m128 tmp2 = _mm_unpacklo_ps(row2.data, row3.data);
        __m128 tmp3 = _mm_unpackhi_ps(row2.data, row3.data);

        row0.data = _mm_movelh_ps(tmp0, tmp2);
        row1.data = _mm_movehl_ps(tmp2, tmp0);
        row2.data = _mm_movelh_ps(tmp1, tmp3);
        row3.data = _mm_movehl_ps(tmp3, tmp1);
    }
};

// AVX (8 floats)
template <> struct simd_vector<float, AVX_SIZE>
{
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<AVX_SIZE>::value, "size must be a power of 2");

    __m256 data; // AVX register

    simd_vector() : data(_mm256_setzero_ps()) {}
    explicit simd_vector(float value) : data(_mm256_set1_ps(value)) {}
    explicit simd_vector(__m256 vec) : data(vec) {}

    float operator[](size_t i) const
    {
        alignas(AVX_ALIGNMENT) std::array<float, AVX_SIZE> elements{};
        _mm256_store_ps(elements.data(), data);
        return elements[i];
    }

    simd_vector &operator+=(const simd_vector &other)
    {
        data = _mm256_add_ps(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector &other) const
    {
        return simd_vector(_mm256_add_ps(data, other.data));
    }

    simd_vector &operator-=(const simd_vector &other)
    {
        data = _mm256_sub_ps(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector &other) const
    {
        return simd_vector(_mm256_sub_ps(data, other.data));
    }

    simd_vector &operator*=(const simd_vector &other)
    {
        data = _mm256_mul_ps(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector &other) const
    {
        return simd_vector(_mm256_mul_ps(data, other.data));
    }

    simd_vector &operator/=(const simd_vector &other)
    {
        data = _mm256_div_ps(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector &other) const
    {
        return simd_vector(_mm256_div_ps(data, other.data));
    }

    // Element-wise conditional operations
    simd_vector operator==(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_EQ_OQ));
    }

    simd_vector operator!=(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_NEQ_OQ));
    }

    simd_vector operator<(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_LT_OQ));
    }

    simd_vector operator<=(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_LE_OQ));
    }

    simd_vector operator>(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_GT_OQ));
    }

    simd_vector operator>=(const simd_vector &other) const
    {
        return simd_vector(_mm256_cmp_ps(data, other.data, _CMP_GE_OQ));
    }

    // Transpose method for 8x8 matrix
    static void transpose(simd_vector &row0, simd_vector &row1, simd_vector &row2,
                          simd_vector &row3, simd_vector &row4, simd_vector &row5,
                          simd_vector &row6, simd_vector &row7)
    {
        __m256 tmp0 = _mm256_unpacklo_ps(row0.data, row1.data);
        __m256 tmp1 = _mm256_unpackhi_ps(row0.data, row1.data);
        __m256 tmp2 = _mm256_unpacklo_ps(row2.data, row3.data);
        __m256 tmp3 = _mm256_unpackhi_ps(row2.data, row3.data);
        __m256 tmp4 = _mm256_unpacklo_ps(row4.data, row5.data);
        __m256 tmp5 = _mm256_unpackhi_ps(row4.data, row5.data);
        __m256 tmp6 = _mm256_unpacklo_ps(row6.data, row7.data);
        __m256 tmp7 = _mm256_unpackhi_ps(row6.data, row7.data);

        row0.data = _mm256_shuffle_ps(tmp0, tmp2, 0x44);
        row1.data = _mm256_shuffle_ps(tmp0, tmp2, 0xEE);
        row2.data = _mm256_shuffle_ps(tmp1, tmp3, 0x44);
        row3.data = _mm256_shuffle_ps(tmp1, tmp3, 0xEE);
        row4.data = _mm256_shuffle_ps(tmp4, tmp6, 0x44);
        row5.data = _mm256_shuffle_ps(tmp4, tmp6, 0xEE);
        row6.data = _mm256_shuffle_ps(tmp5, tmp7, 0x44);
        row7.data = _mm256_shuffle_ps(tmp5, tmp7, 0xEE);
    }
};

// NEON (4 floats, for ARM)
#ifdef __ARM_NEON
template <> struct simd_vector<float, SSE_SIZE>
{
    static_assert(is_supported_type<float>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<SSE_SIZE>::value, "size must be a power of 2");

    float32x4_t data; // NEON register

    simd_vector() : data(vdupq_n_f32(0.0f)) {}
    explicit simd_vector(float value) : data(vdupq_n_f32(value)) {}
    explicit simd_vector(float32x4_t vec) : data(vec) {}

    float operator[](size_t i) const
    {
        std::array<float, SSE_SIZE> elements{};
        vst1q_f32(elements.data(), data);
        return elements[i];
    }

    simd_vector &operator+=(const simd_vector &other)
    {
        data = vaddq_f32(data, other.data);
        return *this;
    }

    simd_vector operator+(const simd_vector &other) const
    {
        return simd_vector(vaddq_f32(data, other.data));
    }

    simd_vector &operator-=(const simd_vector &other)
    {
        data = vsubq_f32(data, other.data);
        return *this;
    }

    simd_vector operator-(const simd_vector &other) const
    {
        return simd_vector(vsubq_f32(data, other.data));
    }

    simd_vector &operator*=(const simd_vector &other)
    {
        data = vmulq_f32(data, other.data);
        return *this;
    }

    simd_vector operator*(const simd_vector &other) const
    {
        return simd_vector(vmulq_f32(data, other.data));
    }

    simd_vector &operator/=(const simd_vector &other)
    {
        data = vdivq_f32(data, other.data);
        return *this;
    }

    simd_vector operator/(const simd_vector &other) const
    {
        return simd_vector(vdivq_f32(data, other.data));
    }

    // Element-wise conditional operations
    simd_vector operator==(const simd_vector &other) const
    {
        return simd_vector(vceqq_f32(data, other.data));
    }

    simd_vector operator!=(const simd_vector &other) const
    {
        return simd_vector(vmvnq_u32(vceqq_f32(data, other.data)));
    }

    simd_vector operator<(const simd_vector &other) const
    {
        return simd_vector(vcltq_f32(data, other.data));
    }

    simd_vector operator<=(const simd_vector &other) const
    {
        return simd_vector(vcleq_f32(data, other.data));
    }

    simd_vector operator>(const simd_vector &other) const
    {
        return simd_vector(vcgtq_f32(data, other.data));
    }

    simd_vector operator>=(const simd_vector &other) const
    {
        return simd_vector(vcgeq_f32(data, other.data));
    }

    // Transpose method for 4x4 matrix
    static void transpose(simd_vector &row0, simd_vector &row1, simd_vector &row2,
                          simd_vector &row3)
    {
        float32x4x2_t tmp0 = vtrnq_f32(row0.data, row1.data);
        float32x4x2_t tmp1 = vtrnq_f32(row2.data, row3.data);

        row0.data = vcombine_f32(vget_low_f32(tmp0.val[0]), vget_low_f32(tmp1.val[0]));
        row1.data = vcombine_f32(vget_low_f32(tmp0.val[1]), vget_low_f32(tmp1.val[1]));
        row2.data = vcombine_f32(vget_high_f32(tmp0.val[0]), vget_high_f32(tmp1.val[0]));
        row3.data = vcombine_f32(vget_high_f32(tmp0.val[1]), vget_high_f32(tmp1.val[1]));
    }
};
#endif

// factory function to create a SIMD vector from a scalar value
template <typename T, size_t N> constexpr simd_vector<T, N> make_vector(T value)
{
    static_assert(is_supported_type<T>::value, "unsupported type for simd_vector");
    static_assert(is_power_of_two<N>::value, "size must be a power of 2");
    return simd_vector<T, N>(value);
}

// SSE
template <> inline simd_vector<float, SSE_SIZE> make_vector<float, SSE_SIZE>(float value)
{
    return simd_vector<float, SSE_SIZE>(value);
}

// AVX
template <> inline simd_vector<float, AVX_SIZE> make_vector<float, AVX_SIZE>(float value)
{
    return simd_vector<float, AVX_SIZE>(value);
}

// NEON
#ifdef __ARM_NEON
template <> inline simd_vector<float, SSE_SIZE> make_vector<float, SSE_SIZE>(float value)
{
    return simd_vector<float, SSE_SIZE>(value);
}
#endif

} // namespace simdlib