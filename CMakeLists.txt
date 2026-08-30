cmake_minimum_required(VERSION 3.24)

# Set the default before CUDA compiler detection; runners detect the actual GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(dscuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter)
enable_testing()

add_library(dscuda_kernels STATIC
    kernels/common/cuda_common.cu
    kernels/attention/attention.cu
    kernels/attention/flash_attention.cu
    kernels/attention/mla.cu
    kernels/attention/rope.cu
    kernels/gemm/matmul.cu
    kernels/gemm/grouped_gemm.cu
    kernels/moe/expert_dispatch.cu
    kernels/normalization/rmsnorm.cu
    kernels/normalization/softmax.cu
    kernels/elementwise/residual.cu
    kernels/elementwise/swiglu.cu
    kernels/embedding/embedding.cu
    kernels/optimizers/adamw.cu
    kernels/optimizers/global_norm.cu
    kernels/losses/cross_entropy.cu
)
target_include_directories(dscuda_kernels PUBLIC include)
target_link_libraries(dscuda_kernels PUBLIC CUDA::cudart)
set_target_properties(dscuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# CPU references belong to correctness tests, not the CUDA library.
add_library(dscuda_reference STATIC
    reference/attention/attention_cpu.cpp
    reference/attention/mla_cpu.cpp
    reference/attention/rope_cpu.cpp
    reference/gemm/matmul_cpu.cpp
    reference/gemm/grouped_gemm_cpu.cpp
    reference/moe/expert_dispatch_cpu.cpp
    reference/normalization/rmsnorm_cpu.cpp
    reference/normalization/softmax_cpu.cpp
    reference/elementwise/residual_cpu.cpp
    reference/elementwise/swiglu_cpu.cpp
    reference/embedding/embedding_cpu.cpp
    reference/optimizers/optimizer_cpu.cpp
    reference/optimizers/global_norm_cpu.cpp
    reference/losses/cross_entropy_cpu.cpp
)
target_include_directories(dscuda_reference PUBLIC include)
target_link_libraries(dscuda_reference PUBLIC CUDA::cudart)
foreach(family attention gemm moe normalization elementwise embedding optimizers losses)
    target_include_directories(dscuda_reference PUBLIC reference/${family})
endforeach()

function(kernel_tests family)
    foreach(name IN LISTS ARGN)
        add_executable(test_${name} tests/${family}/test_${name}.cu)
        target_link_libraries(test_${name} PRIVATE dscuda_kernels dscuda_reference)
        add_test(NAME ${name} COMMAND test_${name})
        set_tests_properties(${name} PROPERTIES LABELS "${family}" RESOURCE_LOCK gpu)
    endforeach()
endfunction()
kernel_tests(attention attention flash_attention mla mla_decode rope)
kernel_tests(gemm matmul grouped_gemm)
kernel_tests(moe expert_dispatch)
kernel_tests(normalization rmsnorm softmax)
kernel_tests(elementwise residual swiglu)
kernel_tests(embedding embedding)
kernel_tests(optimizers adamw global_norm)
kernel_tests(losses cross_entropy)

function(kernel_benchmarks family)
    foreach(name IN LISTS ARGN)
        add_executable(benchmark_${name} benchmarks/${family}/benchmark_${name}.cu)
        target_link_libraries(benchmark_${name} PRIVATE dscuda_kernels)
    endforeach()
endfunction()
kernel_benchmarks(attention attention flash_attention mla mla_decode rope)
kernel_benchmarks(gemm matmul grouped_gemm)
kernel_benchmarks(normalization rmsnorm softmax)
kernel_benchmarks(elementwise swiglu)
kernel_benchmarks(embedding embedding)
kernel_benchmarks(optimizers adamw global_norm)
kernel_benchmarks(losses cross_entropy)
target_link_libraries(benchmark_matmul PRIVATE CUDA::cublas)
target_link_libraries(benchmark_grouped_gemm PRIVATE CUDA::cublas)

# ctypes bridges keep custom and library calls in the same Python/CUDA process.
add_library(dscuda_flash_attention_bench SHARED benchmarks/bindings/flash_attention_bridge.cu)
target_link_libraries(dscuda_flash_attention_bench PRIVATE dscuda_kernels)
add_library(dscuda_operator_bench SHARED benchmarks/bindings/operators_bridge.cu)
target_link_libraries(dscuda_operator_bench PRIVATE dscuda_kernels CUDA::cublas)
add_library(dscuda_mla_bench SHARED benchmarks/bindings/mla_bridge.cu)
target_link_libraries(dscuda_mla_bench PRIVATE dscuda_kernels)

if(Python3_Interpreter_FOUND)
    add_test(NAME mla_benchmark_python
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/attention/test_mla_benchmark.py)
    add_test(NAME operator_benchmark_python
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_operator_benchmark.py)
    add_test(NAME attention_benchmark_python
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/attention/test_attention_benchmark.py)
endif()

# Optional PyTorch checks use the uv environment, not the system Python.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    add_test(NAME mla_reference_python
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/attention/test_mla_reference.py)
    add_test(NAME mla_pytorch_cuda
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_mla_runtime.py
        --library $<TARGET_FILE:dscuda_mla_bench> --check-only)
    set_tests_properties(mla_pytorch_cuda PROPERTIES LABELS "attention" RESOURCE_LOCK gpu)
endif()
