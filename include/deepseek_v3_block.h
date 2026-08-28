#pragma once

#include "deepseek_moe.h"
#include "mla.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct DeepSeekV3BlockConfig {
    MlaLayerConfig mla;
    DeepSeekMoeConfig moe;
    float epsilon;
};

struct DeepSeekV3BlockParameters {
    const float* attention_norm_weight;
    MlaLayerParameters mla;
    const float* ffn_norm_weight;
    DeepSeekMoeParameters moe;
};

struct DeepSeekV3BlockGradients {
    float* attention_norm_weight;
    MlaLayerGradients mla;
    float* ffn_norm_weight;
    DeepSeekMoeGradients moe;
};

std::size_t deepseek_v3_block_activation_elements(
    const DeepSeekV3BlockConfig& config);
std::size_t deepseek_v3_block_integer_activation_elements(
    const DeepSeekV3BlockConfig& config);
std::size_t deepseek_v3_block_backward_workspace_elements(
    const DeepSeekV3BlockConfig& config);
std::size_t deepseek_v3_block_bf16_workspace_elements(
    const DeepSeekV3BlockConfig& config);

void deepseek_v3_block_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    int* integer_activations,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream = nullptr);

void deepseek_v3_block_backward_cuda(
    float* input_gradient,
    const DeepSeekV3BlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream = nullptr);

void deepseek_v3_block_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const DeepSeekV3BlockConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
