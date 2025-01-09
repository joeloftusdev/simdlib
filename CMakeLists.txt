cmake_minimum_required(VERSION 3.10)

project(SIMDLibrary)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_library(simdlib STATIC src/simd_vector.cpp)

add_executable(main main.cpp)
target_link_libraries(main simdlib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES *.cpp *.hpp)
    add_custom_target(
        clang-format
        COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_SOURCE_FILES}
        COMMENT "Running clang-format"
    )
endif()

include(gtests)
include(benchmark)