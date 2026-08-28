#pragma once

#include "deepseek_v3_block.h"

namespace dscuda {

void deepseek_v3_block_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3BlockConfig& config);

void deepseek_v3_block_backward_cpu(
    float* input_gradient,
    const DeepSeekV3BlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3BlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3BlockConfig& config);

}  // namespace dscuda
