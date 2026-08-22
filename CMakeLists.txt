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

add_library(dscuda_kernels STATIC
    kernels/cuda_common.cu
    kernels/matmul.cu
    kernels/rmsnorm.cu
    kernels/swiglu.cu
    reference/matmul_cpu.cpp
    reference/rmsnorm_cpu.cpp
    reference/rope_cpu.cpp
    reference/softmax_cpu.cpp
    reference/swiglu_cpu.cpp
)

target_include_directories(dscuda_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/reference
)

target_link_libraries(dscuda_kernels PUBLIC CUDA::cudart)
set_target_properties(dscuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_rmsnorm tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE dscuda_kernels)

add_executable(benchmark_rmsnorm benchmarks/benchmark_rmsnorm.cu)
target_link_libraries(benchmark_rmsnorm PRIVATE dscuda_kernels)

add_executable(test_swiglu tests/test_swiglu.cu)
target_link_libraries(test_swiglu PRIVATE dscuda_kernels)

add_executable(benchmark_swiglu benchmarks/benchmark_swiglu.cu)
target_link_libraries(benchmark_swiglu PRIVATE dscuda_kernels)

add_executable(test_matmul tests/test_matmul.cu)
target_link_libraries(test_matmul PRIVATE dscuda_kernels)

add_executable(benchmark_matmul benchmarks/benchmark_matmul.cu)
target_link_libraries(benchmark_matmul PRIVATE dscuda_kernels CUDA::cublas)

enable_testing()
add_test(NAME rmsnorm COMMAND test_rmsnorm)
add_test(NAME swiglu COMMAND test_swiglu)
add_test(NAME matmul COMMAND test_matmul)
