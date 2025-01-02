cmake_minimum_required(VERSION 3.10)

project(SIMDLibrary)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_library(simdlib STATIC src/simd_vector.cpp)

add_executable(main main.cpp)
target_link_libraries(main simdlib)
