#include <benchmark/benchmark.h>
#include "../include/simdlib/simd_vector.hpp"

static void BM_SimdVectorAddition(benchmark::State& state) {
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vec1 += vec2);
    }
}
BENCHMARK(BM_SimdVectorAddition);

static void BM_SimdVectorSubtraction(benchmark::State& state) {
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vec1 -= vec2);
    }
}
BENCHMARK(BM_SimdVectorSubtraction);

static void BM_SimdVectorMultiplication(benchmark::State& state) {
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vec1 *= vec2);
    }
}
BENCHMARK(BM_SimdVectorMultiplication);

static void BM_SimdVectorDivision(benchmark::State& state) {
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state) {
        benchmark::DoNotOptimize(vec1 /= vec2);
    }
}
BENCHMARK(BM_SimdVectorDivision);

BENCHMARK_MAIN();