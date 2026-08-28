#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

struct DeepSeekMoeConfig {
    int rows;
    int hidden_size;
    int expert_hidden_size;
    int routed_experts;
    int shared_experts;
    int top_k;
    float route_scale;
    int batch_size = 1;
    int sequence_length = 0;
    float balance_loss_weight = 0.0F;
};

struct DeepSeekMoeParameters {
    const float* router_weight;
    const float* routing_bias;
    const float* routed_gate_weight;
    const float* routed_up_weight;
    const float* routed_down_weight;
    const float* shared_gate_weight;
    const float* shared_up_weight;
    const float* shared_down_weight;
};

struct DeepSeekMoeGradients {
    float* router_weight;
    float* routed_gate_weight;
    float* routed_up_weight;
    float* routed_down_weight;
    float* shared_gate_weight;
    float* shared_up_weight;
    float* shared_down_weight;
};

std::size_t deepseek_moe_activation_elements(
    const DeepSeekMoeConfig& config);
std::size_t deepseek_moe_integer_activation_elements(
    const DeepSeekMoeConfig& config);
std::size_t deepseek_moe_backward_workspace_elements(
    const DeepSeekMoeConfig& config);

void deepseek_moe_forward_cuda(
    float* output,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    float* activations,
    int* integer_activations,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream = nullptr);

void deepseek_moe_backward_cuda(
    float* input_gradient,
    const DeepSeekMoeGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const float* activations,
    const int* integer_activations,
    float* workspace,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream = nullptr);

// Applies DeepSeek-V3's per-step overloaded/underloaded expert bias update.
void deepseek_moe_update_bias_cuda(
    float* routing_bias,
    const int* integer_activations,
    const DeepSeekMoeConfig& config,
    float update_speed,
    cudaStream_t stream = nullptr);

// Adds the complementary sequence-wise router-balance objective to a device
// scalar. The selected loads use unbiased affinities, matching V3 equation 17.
void deepseek_moe_add_balance_loss_cuda(
    float* total_loss,
    const float* activations,
    const DeepSeekMoeConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
