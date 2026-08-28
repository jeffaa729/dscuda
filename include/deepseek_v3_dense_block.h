#pragma once

#include "mla.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct DeepSeekV3DenseBlockConfig {
    MlaLayerConfig mla;
    int rows;
    int hidden_size;
    int ffn_size;
    float epsilon;
};

struct DeepSeekV3DenseBlockParameters {
    const float* attention_norm_weight;
    MlaLayerParameters mla;
    const float* ffn_norm_weight;
    const float* gate_weight;
    const float* up_weight;
    const float* down_weight;
};

struct DeepSeekV3DenseBlockGradients {
    float* attention_norm_weight;
    MlaLayerGradients mla;
    float* ffn_norm_weight;
    float* gate_weight;
    float* up_weight;
    float* down_weight;
};

std::size_t deepseek_v3_dense_block_activation_elements(
    const DeepSeekV3DenseBlockConfig& config);
std::size_t deepseek_v3_dense_block_backward_workspace_elements(
    const DeepSeekV3DenseBlockConfig& config);
std::size_t deepseek_v3_dense_block_bf16_workspace_elements(
    const DeepSeekV3DenseBlockConfig& config);

void deepseek_v3_dense_block_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3DenseBlockConfig& config,
    cudaStream_t stream = nullptr);

void deepseek_v3_dense_block_backward_cuda(
    float* input_gradient,
    const DeepSeekV3DenseBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3DenseBlockConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
