#include <gtest/gtest.h>
#include "../include/simdlib/simd_vector.hpp"
#include "../include/simdlib/simd_operations.hpp"
#include <iostream>

namespace simdlib {

TEST(SimdVectorTest, Initialization) {
    simd_vector<float, 4> vec1(1.0f);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vec1[i], 1.0f);
    }

    simd_vector<float, 8> vec2(2.0f);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(vec2[i], 2.0f);
    }
}

TEST(SimdVectorTest, Addition) {
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 + vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 3.0f);
    }
}

TEST(SimdVectorTest, AdditionInPlace) {
    simd_vector<float, 4> vec1(1.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 += vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vec1[i], 3.0f);
    }
}

TEST(SimdVectorTest, Subtraction) {
    simd_vector<float, 4> vec1(3.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 - vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 1.0f);
    }
}

TEST(SimdVectorTest, SubtractionInPlace) {
    simd_vector<float, 4> vec1(3.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 -= vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vec1[i], 1.0f);
    }
}

TEST(SimdVectorTest, Multiplication) {
    simd_vector<float, 4> vec1(2.0f);
    simd_vector<float, 4> vec2(3.0f);
    auto result = vec1 * vec2;


    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 6.0f);
    }
}

TEST(SimdVectorTest, MultiplicationInPlace) {
    simd_vector<float, 4> vec1(2.0f);
    simd_vector<float, 4> vec2(3.0f);
    vec1 *= vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vec1[i], 6.0f);
    }
}

TEST(SimdVectorTest, Division) {
    simd_vector<float, 4> vec1(6.0f);
    simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 / vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[i], 3.0f);
    }
}

TEST(SimdVectorTest, DivisionInPlace) {
    simd_vector<float, 4> vec1(6.0f);
    simd_vector<float, 4> vec2(2.0f);
    vec1 /= vec2;

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(vec1[i], 3.0f);
    }
}

} // namespace simdlib

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}