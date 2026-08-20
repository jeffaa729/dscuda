cmake_minimum_required(VERSION 3.24)

project(dscuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(dscuda_rmsnorm STATIC
    kernels/cuda_common.cu
    kernels/rmsnorm.cu
    reference/rmsnorm_cpu.cpp
)

target_include_directories(dscuda_rmsnorm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/reference
)

target_link_libraries(dscuda_rmsnorm PUBLIC CUDA::cudart)
set_target_properties(dscuda_rmsnorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_rmsnorm tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE dscuda_rmsnorm)

add_executable(benchmark_rmsnorm benchmarks/benchmark_rmsnorm.cu)
target_link_libraries(benchmark_rmsnorm PRIVATE dscuda_rmsnorm)

enable_testing()
add_test(NAME rmsnorm COMMAND test_rmsnorm)
