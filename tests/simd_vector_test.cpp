#include <gtest/gtest.h>
#include "../include/simdlib/simd_vector.hpp"
#include "../include/simdlib/simd_operations.hpp"
#include <iostream>

namespace simdlib
{

TEST(SimdVectorTest, Initialization)
{
    simd_vector<float, 4> vec1(1.0f);
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(vec1[i], 1.0f);
    }

    simd_vector<float, 8> vec2(2.0f);
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(vec2[i], 2.0f);
    }
}

TEST(SimdVectorTest, Addition)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 + vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(result[i], 3.0f);
    }
}

TEST(SimdVectorTest, AdditionInPlace)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 += vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(vec1[i], 3.0f);
    }
}

TEST(SimdVectorTest, Subtraction)
{
    simd_vector<float, 4> vec1(3.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 - vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(result[i], 1.0f);
    }
}

TEST(SimdVectorTest, SubtractionInPlace)
{
    simd_vector<float, 4> vec1(3.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 -= vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(vec1[i], 1.0f);
    }
}

TEST(SimdVectorTest, Multiplication)
{
    simd_vector<float, 4> vec1(2.0f);
    simd_vector<float, 4> vec2(3.0f);
    auto result = vec1 * vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(result[i], 6.0f);
    }
}

TEST(SimdVectorTest, MultiplicationInPlace)
{
    simd_vector<float, 4> vec1(2.0f);
    simd_vector<float, 4> vec2(3.0f);
    vec1 *= vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(vec1[i], 6.0f);
    }
}

TEST(SimdVectorTest, Division)
{
    simd_vector<float, 4> vec1(6.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 / vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(result[i], 3.0f);
    }
}

TEST(SimdVectorTest, DivisionInPlace)
{
    simd_vector<float, 4> vec1(6.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 /= vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(vec1[i], 3.0f);
    }
}

TEST(SimdVectorTest, Equality)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(1.0f);
    auto result = vec1 == vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF); // 0xFFFFFFFF = true
    }
}

TEST(SimdVectorTest, Inequality)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 != vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF);
    }
}

TEST(SimdVectorTest, LessThan)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 < vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF);
    }
}

TEST(SimdVectorTest, LessThanOrEqual)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(1.0f);
    auto result = vec1 <= vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF);
    }
}

TEST(SimdVectorTest, GreaterThan)
{
    simd_vector<float, 4> vec1(2.0f);
    simd_vector<float, 4> vec2(1.0f);
    auto result = vec1 > vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF);
    }
}

TEST(SimdVectorTest, GreaterThanOrEqual)
{
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(1.0f);
    auto result = vec1 >= vec2;

    for (size_t i = 0; i < 4; ++i)
    {
        float temp = result[i];
        uint32_t mask = reinterpret_cast<const uint32_t &>(temp);
        EXPECT_EQ(mask, 0xFFFFFFFF);
    }
}

TEST(SimdVectorTest, HorizontalSum)
{
    simd_vector<float, 4> vec(1.0f, 2.0f, 3.0f, 4.0f);
    float result = vec.horizontal_sum();
    EXPECT_EQ(result, 10.0f);
}

TEST(SimdVectorTest, HorizontalMax)
{
    simd_vector<float, 4> vec(1.0f, 2.0f, 3.0f, 4.0f);
    float result = vec.horizontal_max();
    EXPECT_EQ(result, 4.0f);
}

TEST(SimdVectorTest, HorizontalMin)
{
    simd_vector<float, 4> vec(1.0f, 2.0f, 3.0f, 4.0f);
    float result = vec.horizontal_min();
    EXPECT_EQ(result, 1.0f);
}

TEST(SimdVectorTest, Shuffle)
{
    simd_vector<float, 4> vec(1.0f, 2.0f, 3.0f, 4.0f);
    auto result = vec.shuffle(_MM_SHUFFLE(0, 1, 2, 3));

    EXPECT_EQ(result[0], 4.0f);
    EXPECT_EQ(result[1], 3.0f);
    EXPECT_EQ(result[2], 2.0f);
    EXPECT_EQ(result[3], 1.0f);
}

TEST(SimdVectorTest, Permute)
{
    simd_vector<float, 4> vec(1.0f, 2.0f, 3.0f, 4.0f);
    auto result = vec.permute(_MM_SHUFFLE(2, 3, 0, 1));

    EXPECT_EQ(result[0], 2.0f);
    EXPECT_EQ(result[1], 1.0f);
    EXPECT_EQ(result[2], 4.0f);
    EXPECT_EQ(result[3], 3.0f);
}

TEST(SimdVectorTest, Blend)
{
    simd_vector<float, 4> vec1(1.0f, 2.0f, 3.0f, 4.0f);
    simd_vector<float, 4> vec2(5.0f, 6.0f, 7.0f, 8.0f);
    auto result = vec1.blend(vec2, 0b1010);

    EXPECT_EQ(result[0], 1.0f);
    EXPECT_EQ(result[1], 6.0f);
    EXPECT_EQ(result[2], 3.0f);
    EXPECT_EQ(result[3], 8.0f);
}

} // namespace simdlib

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}