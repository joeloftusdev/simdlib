#include <iostream>
#include "include/simdlib/simd_vector.hpp"
#include "include/simdlib/simd_operations.hpp"
#include "include/simdlib/simd_utils.hpp"

int main() {
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    auto result = vec1 + vec2;

    std::cout << "Result: " << result << std::endl; // Result: [3, 3, 3, 3]

    return 0;
}