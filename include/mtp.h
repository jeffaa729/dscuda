#pragma once

#include "deepseek_v3_block.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct MtpConfig {
    int batch_size;
    int sequence_length;
    int hidden_size;
    int depth;
    float epsilon;
    DeepSeekV3BlockConfig block;
};

struct MtpParameters {
    const float* hidden_norm_weight;
    const float* embedding_norm_weight;
    const float* projection_weight;
    DeepSeekV3BlockParameters block;
};

struct MtpGradients {
    float* hidden_norm_weight;
    float* embedding_norm_weight;
    float* projection_weight;
    DeepSeekV3BlockGradients block;
};

std::size_t mtp_activation_elements(const MtpConfig& config);
std::size_t mtp_integer_activation_elements(const MtpConfig& config);
std::size_t mtp_backward_workspace_elements(const MtpConfig& config);
std::size_t mtp_bf16_workspace_elements(const MtpConfig& config);

void mtp_forward_cuda(
    float* output,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    float* activations,
    int* integer_activations,
    __nv_bfloat16* bf16_workspace,
    const MtpConfig& config,
    cudaStream_t stream = nullptr);

void mtp_backward_cuda(
    float* previous_hidden_gradient,
    float* embedding_gradient,
    const MtpGradients& parameter_gradients,
    const float* output_gradient,
    const float* previous_hidden,
    const int* tokens,
    const float* embedding_table,
    const MtpParameters& parameters,
    const float* cosine,
    const float* sine,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    __nv_bfloat16* bf16_workspace,
    const MtpConfig& config,
    cudaStream_t stream = nullptr);

void mtp_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const MtpConfig& config,
    cudaStream_t stream = nullptr);

void mtp_gather_targets_cuda(
    int* shifted_targets,
    const int* next_token_targets,
    const MtpConfig& config,
    cudaStream_t stream = nullptr);

void mtp_add_scaled_loss_cuda(
    float* total_loss,
    const float* module_loss,
    float scale,
    cudaStream_t stream = nullptr);

void mtp_scale_gradients_cuda(
    float* gradients,
    int elements,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
