#pragma once

#include "deepseek_moe.h"

namespace dscuda {

void deepseek_moe_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config);

void deepseek_moe_backward_cpu(
    float* input_gradient,
    const DeepSeekMoeGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config);

float deepseek_moe_balance_loss_cpu(
    const float* input,
    const DeepSeekMoeParameters& parameters,
    const DeepSeekMoeConfig& config);

}  // namespace dscuda
