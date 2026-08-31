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
enable_testing()

add_library(dscuda_kernels STATIC
    kernels/common/cuda_common.cu
    kernels/attention/flash_attention.cu
    kernels/attention/mla.cu
    kernels/gemm/matmul.cu
    kernels/gemm/grouped_gemm.cu
    kernels/moe/expert_dispatch.cu
)
target_include_directories(dscuda_kernels PUBLIC include)
target_link_libraries(dscuda_kernels PUBLIC CUDA::cudart)
set_target_properties(dscuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Optional Nsight launchers. Normal tests/runtime benchmarks use the Python runner.
function(kernel_profilers family)
    foreach(name IN LISTS ARGN)
        add_executable(benchmark_${name} EXCLUDE_FROM_ALL benchmarks/${family}/benchmark_${name}.cu)
        target_link_libraries(benchmark_${name} PRIVATE dscuda_kernels)
    endforeach()
endfunction()
kernel_profilers(attention flash_attention mla mla_decode)
kernel_profilers(gemm matmul grouped_gemm)
target_link_libraries(benchmark_matmul PRIVATE CUDA::cublas)
target_link_libraries(benchmark_grouped_gemm PRIVATE CUDA::cublas)

# ctypes bridges keep custom and library calls in the same Python/CUDA process.
add_library(dscuda_flash_attention_bench SHARED benchmarks/bindings/flash_attention_bridge.cu)
target_link_libraries(dscuda_flash_attention_bench PRIVATE dscuda_kernels)
add_library(dscuda_operator_bench SHARED benchmarks/bindings/operators_bridge.cu)
target_link_libraries(dscuda_operator_bench PRIVATE dscuda_kernels CUDA::cublas)
add_library(dscuda_mla_bench SHARED benchmarks/bindings/mla_bridge.cu)
target_link_libraries(dscuda_mla_bench PRIVATE dscuda_kernels)
add_library(dscuda_expert_dispatch_bench SHARED benchmarks/bindings/expert_dispatch_bridge.cu)
target_link_libraries(dscuda_expert_dispatch_bench PRIVATE dscuda_kernels)

# Tests and benchmarks share cases and compare numerical results with PyTorch.
set(DSCUDA_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python from uv sync")
if(EXISTS "${DSCUDA_PYTHON}")
    foreach(name matmul grouped_gemm flash_attention mla expert_dispatch kda)
        add_test(NAME ${name} COMMAND ${DSCUDA_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/run.py ${name} --test)
        set_tests_properties(${name} PROPERTIES RESOURCE_LOCK gpu)
    endforeach()
endif()
