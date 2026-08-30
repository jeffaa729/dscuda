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
find_package(Python3 COMPONENTS Interpreter)

add_library(dscuda_data STATIC
    src/dataset.cpp
    src/tokenizer.cpp
)
target_include_directories(dscuda_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dscuda_kernels STATIC
    src/checkpoint.cpp
    src/generation.cpp
    kernels/attention/attention.cu
    kernels/attention/flash_attention.cu
    kernels/attention/mla.cu
    kernels/attention/rope.cu
    kernels/common/cuda_common.cu
    kernels/elementwise/residual.cu
    kernels/elementwise/swiglu.cu
    kernels/embedding/embedding.cu
    kernels/matmul/matmul.cu
    kernels/model/deepseek_v3_block.cu
    kernels/model/deepseek_v3_dense_block.cu
    kernels/model/deepseek_v3_model.cu
    kernels/model/model.cu
    kernels/model/mtp.cu
    kernels/model/transformer_block.cu
    kernels/moe/deepseek_moe.cu
    kernels/moe/expert_dispatch.cu
    kernels/normalization/rmsnorm.cu
    kernels/normalization/softmax.cu
    kernels/training/adamw.cu
    kernels/training/cross_entropy.cu
    kernels/training/global_norm.cu
    reference/attention_cpu.cpp
    reference/cross_entropy_cpu.cpp
    reference/deepseek_moe_cpu.cpp
    reference/deepseek_v3_block_cpu.cpp
    reference/deepseek_v3_dense_block_cpu.cpp
    reference/embedding_cpu.cpp
    reference/expert_dispatch_cpu.cpp
    reference/global_norm_cpu.cpp
    reference/matmul_cpu.cpp
    reference/mla_cpu.cpp
    reference/model_cpu.cpp
    reference/mtp_cpu.cpp
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

add_executable(test_flash_attention tests/test_flash_attention.cu)
target_link_libraries(test_flash_attention PRIVATE dscuda_kernels)

add_executable(test_mla tests/test_mla.cu)
target_link_libraries(test_mla PRIVATE dscuda_kernels)

add_executable(test_mla_decode tests/test_mla_decode.cu)
target_link_libraries(test_mla_decode PRIVATE dscuda_kernels)

add_executable(test_mla_layer_decode tests/test_mla_layer_decode.cu)
target_link_libraries(test_mla_layer_decode PRIVATE dscuda_kernels)

add_executable(test_mla_layer tests/test_mla_layer.cu)
target_link_libraries(test_mla_layer PRIVATE dscuda_kernels)

add_executable(benchmark_mla benchmarks/benchmark_mla.cu)
target_link_libraries(benchmark_mla PRIVATE dscuda_kernels)

add_executable(benchmark_mla_decode benchmarks/benchmark_mla_decode.cu)
target_link_libraries(benchmark_mla_decode PRIVATE dscuda_kernels)

add_executable(benchmark_attention benchmarks/benchmark_attention.cu)
target_link_libraries(benchmark_attention PRIVATE dscuda_kernels)

add_executable(benchmark_flash_attention benchmarks/benchmark_flash_attention.cu)
target_link_libraries(benchmark_flash_attention PRIVATE dscuda_kernels)

# Compile just the attention operator into a small shared bridge. The ordinary
# training library stays unchanged; no PyTorch extension toolchain is needed.
add_library(dscuda_flash_attention_bench SHARED
    benchmarks/reference/flash_attention_bridge.cu
    kernels/attention/flash_attention.cu
    kernels/common/cuda_common.cu
)
target_include_directories(dscuda_flash_attention_bench PRIVATE include)
target_link_libraries(dscuda_flash_attention_bench PRIVATE CUDA::cudart)
set_target_properties(dscuda_flash_attention_bench PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_executable(benchmark_training_step benchmarks/benchmark_training_step.cu)
target_link_libraries(benchmark_training_step PRIVATE dscuda_kernels)

add_executable(test_residual tests/test_residual.cu)
target_link_libraries(test_residual PRIVATE dscuda_kernels)

add_executable(test_transformer_block tests/test_transformer_block.cu)
target_link_libraries(test_transformer_block PRIVATE dscuda_kernels)

add_executable(benchmark_transformer_block benchmarks/benchmark_transformer_block.cu)
target_link_libraries(benchmark_transformer_block PRIVATE dscuda_kernels)

add_executable(test_embedding tests/test_embedding.cu)
target_link_libraries(test_embedding PRIVATE dscuda_kernels)

add_executable(test_expert_dispatch tests/test_expert_dispatch.cu)
target_link_libraries(test_expert_dispatch PRIVATE dscuda_kernels)

add_executable(test_deepseek_moe tests/test_deepseek_moe.cu)
target_link_libraries(test_deepseek_moe PRIVATE dscuda_kernels)

add_executable(test_grouped_gemm tests/test_grouped_gemm.cu)
target_link_libraries(test_grouped_gemm PRIVATE dscuda_kernels)

add_executable(benchmark_moe benchmarks/benchmark_moe.cu)
target_link_libraries(benchmark_moe PRIVATE dscuda_kernels CUDA::cublas)

add_executable(test_deepseek_v3_block tests/test_deepseek_v3_block.cu)
target_link_libraries(test_deepseek_v3_block PRIVATE dscuda_kernels)

add_executable(
    test_deepseek_v3_dense_block
    tests/test_deepseek_v3_dense_block.cu)
target_link_libraries(test_deepseek_v3_dense_block PRIVATE dscuda_kernels)

add_executable(test_deepseek_v3_model tests/test_deepseek_v3_model.cu)
target_link_libraries(test_deepseek_v3_model PRIVATE dscuda_kernels)

add_executable(test_mtp tests/test_mtp.cu)
target_link_libraries(test_mtp PRIVATE dscuda_kernels)

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

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE dscuda_data)

add_executable(test_dataset tests/test_dataset.cpp)
target_link_libraries(test_dataset PRIVATE dscuda_data)

add_executable(test_model tests/test_model.cu)
target_link_libraries(test_model PRIVATE dscuda_kernels)

add_executable(test_model_bf16 tests/test_model_bf16.cu)
target_link_libraries(test_model_bf16 PRIVATE dscuda_kernels)

add_executable(train_dscuda train_dscuda.cu)
target_link_libraries(train_dscuda PRIVATE dscuda_kernels dscuda_data)

add_executable(generate_dscuda generate_dscuda.cu)
target_link_libraries(generate_dscuda PRIVATE dscuda_kernels dscuda_data)

add_executable(test_checkpoint tests/test_checkpoint.cu)
target_link_libraries(test_checkpoint PRIVATE dscuda_kernels)

add_executable(test_generation tests/test_generation.cpp)
target_link_libraries(test_generation PRIVATE dscuda_kernels)

enable_testing()
add_test(NAME rmsnorm COMMAND test_rmsnorm)
add_test(NAME swiglu COMMAND test_swiglu)
add_test(NAME matmul COMMAND test_matmul)
add_test(NAME rope COMMAND test_rope)
add_test(NAME softmax COMMAND test_softmax)
add_test(NAME attention COMMAND test_attention)
add_test(NAME flash_attention COMMAND test_flash_attention)
add_test(NAME mla COMMAND test_mla)
add_test(NAME mla_decode COMMAND test_mla_decode)
add_test(NAME mla_layer_decode COMMAND test_mla_layer_decode)
add_test(NAME mla_layer COMMAND test_mla_layer)
add_test(NAME residual COMMAND test_residual)
add_test(NAME transformer_block COMMAND test_transformer_block)
add_test(NAME embedding COMMAND test_embedding)
add_test(NAME expert_dispatch COMMAND test_expert_dispatch)
add_test(NAME deepseek_moe COMMAND test_deepseek_moe)
add_test(NAME grouped_gemm COMMAND test_grouped_gemm)
add_test(NAME deepseek_v3_block COMMAND test_deepseek_v3_block)
add_test(NAME deepseek_v3_dense_block COMMAND test_deepseek_v3_dense_block)
add_test(NAME deepseek_v3_model COMMAND test_deepseek_v3_model)
add_test(NAME mtp COMMAND test_mtp)
add_test(NAME adamw COMMAND test_adamw)
add_test(NAME cross_entropy COMMAND test_cross_entropy)
add_test(NAME global_norm COMMAND test_global_norm)
add_test(NAME tokenizer_cpp COMMAND test_tokenizer)
add_test(NAME dataset COMMAND test_dataset)
add_test(NAME model COMMAND test_model)
add_test(NAME model_bf16 COMMAND test_model_bf16)
add_test(NAME checkpoint COMMAND test_checkpoint)
add_test(NAME generation COMMAND test_generation)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME attention_benchmark_python
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_attention_benchmark.py
    )
    add_test(
        NAME tokenizer_python
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tokenizer.py
    )
endif()
