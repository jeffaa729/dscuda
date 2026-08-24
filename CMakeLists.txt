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
    kernels/adamw.cu
    kernels/attention.cu
    kernels/cross_entropy.cu
    kernels/cuda_common.cu
    kernels/embedding.cu
    kernels/global_norm.cu
    kernels/matmul.cu
    kernels/rmsnorm.cu
    kernels/residual.cu
    kernels/rope.cu
    kernels/softmax.cu
    kernels/swiglu.cu
    kernels/transformer_block.cu
    reference/attention_cpu.cpp
    reference/cross_entropy_cpu.cpp
    reference/embedding_cpu.cpp
    reference/global_norm_cpu.cpp
    reference/matmul_cpu.cpp
    reference/rmsnorm_cpu.cpp
    reference/optimizer_cpu.cpp
    reference/residual_cpu.cpp
    reference/rope_cpu.cpp
    reference/softmax_cpu.cpp
    reference/swiglu_cpu.cpp
    reference/transformer_block_cpu.cpp
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

add_executable(test_rope tests/test_rope.cu)
target_link_libraries(test_rope PRIVATE dscuda_kernels)

add_executable(benchmark_rope benchmarks/benchmark_rope.cu)
target_link_libraries(benchmark_rope PRIVATE dscuda_kernels)

add_executable(test_softmax tests/test_softmax.cu)
target_link_libraries(test_softmax PRIVATE dscuda_kernels)

add_executable(benchmark_softmax benchmarks/benchmark_softmax.cu)
target_link_libraries(benchmark_softmax PRIVATE dscuda_kernels)

add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention PRIVATE dscuda_kernels)

add_executable(benchmark_attention benchmarks/benchmark_attention.cu)
target_link_libraries(benchmark_attention PRIVATE dscuda_kernels)

add_executable(test_residual tests/test_residual.cu)
target_link_libraries(test_residual PRIVATE dscuda_kernels)

add_executable(test_transformer_block tests/test_transformer_block.cu)
target_link_libraries(test_transformer_block PRIVATE dscuda_kernels)

add_executable(benchmark_transformer_block benchmarks/benchmark_transformer_block.cu)
target_link_libraries(benchmark_transformer_block PRIVATE dscuda_kernels)

add_executable(test_embedding tests/test_embedding.cu)
target_link_libraries(test_embedding PRIVATE dscuda_kernels)

add_executable(benchmark_embedding benchmarks/benchmark_embedding.cu)
target_link_libraries(benchmark_embedding PRIVATE dscuda_kernels)

add_executable(test_adamw tests/test_adamw.cu)
target_link_libraries(test_adamw PRIVATE dscuda_kernels)

add_executable(benchmark_adamw benchmarks/benchmark_adamw.cu)
target_link_libraries(benchmark_adamw PRIVATE dscuda_kernels)

add_executable(test_cross_entropy tests/test_cross_entropy.cu)
target_link_libraries(test_cross_entropy PRIVATE dscuda_kernels)

add_executable(test_global_norm tests/test_global_norm.cu)
target_link_libraries(test_global_norm PRIVATE dscuda_kernels)

add_executable(benchmark_cross_entropy benchmarks/benchmark_cross_entropy.cu)
target_link_libraries(benchmark_cross_entropy PRIVATE dscuda_kernels)

add_executable(benchmark_global_norm benchmarks/benchmark_global_norm.cu)
target_link_libraries(benchmark_global_norm PRIVATE dscuda_kernels)

enable_testing()
add_test(NAME rmsnorm COMMAND test_rmsnorm)
add_test(NAME swiglu COMMAND test_swiglu)
add_test(NAME matmul COMMAND test_matmul)
add_test(NAME rope COMMAND test_rope)
add_test(NAME softmax COMMAND test_softmax)
add_test(NAME attention COMMAND test_attention)
add_test(NAME residual COMMAND test_residual)
add_test(NAME transformer_block COMMAND test_transformer_block)
add_test(NAME embedding COMMAND test_embedding)
add_test(NAME adamw COMMAND test_adamw)
add_test(NAME cross_entropy COMMAND test_cross_entropy)
add_test(NAME global_norm COMMAND test_global_norm)
