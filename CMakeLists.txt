cmake_minimum_required(VERSION 3.10)

project(SIMDLibrary)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_library(simdlib STATIC src/simd_vector.cpp)

add_executable(main main.cpp)
target_link_libraries(main simdlib)

include(gtests)