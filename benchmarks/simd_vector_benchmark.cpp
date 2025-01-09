#include <benchmark/benchmark.h>
#include "../include/simdlib/simd_vector.hpp"

static void BM_SimdVectorAddition(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 += vec2);
    }
}
BENCHMARK(BM_SimdVectorAddition);

static void BM_SimdVectorSubtraction(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 -= vec2);
    }
}
BENCHMARK(BM_SimdVectorSubtraction);

static void BM_SimdVectorMultiplication(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 *= vec2);
    }
}
BENCHMARK(BM_SimdVectorMultiplication);

static void BM_SimdVectorDivision(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 /= vec2);
    }
}
BENCHMARK(BM_SimdVectorDivision);

static void BM_SimdVectorEquality(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(1.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 == vec2);
    }
}
BENCHMARK(BM_SimdVectorEquality);

static void BM_SimdVectorInequality(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 != vec2);
    }
}
BENCHMARK(BM_SimdVectorInequality);

static void BM_SimdVectorLessThan(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(2.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 < vec2);
    }
}
BENCHMARK(BM_SimdVectorLessThan);

static void BM_SimdVectorLessThanOrEqual(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(1.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 <= vec2);
    }
}
BENCHMARK(BM_SimdVectorLessThanOrEqual);

static void BM_SimdVectorGreaterThan(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(2.0f);
    simdlib::simd_vector<float, 4> vec2(1.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 > vec2);
    }
}
BENCHMARK(BM_SimdVectorGreaterThan);

static void BM_SimdVectorGreaterThanOrEqual(benchmark::State &state)
{
    simdlib::simd_vector<float, 4> vec1(1.0f);
    simdlib::simd_vector<float, 4> vec2(1.0f);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec1 >= vec2);
    }
}
BENCHMARK(BM_SimdVectorGreaterThanOrEqual);

BENCHMARK_MAIN();