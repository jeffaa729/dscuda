#pragma once

#include "deepseek_v3_dense_block.h"

namespace dscuda {

void deepseek_v3_dense_block_forward_cpu(
    float* output,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3DenseBlockConfig& config);

void deepseek_v3_dense_block_backward_cpu(
    float* input_gradient,
    const DeepSeekV3DenseBlockGradients& parameter_gradients,
    const float* output_gradient,
    const float* input,
    const DeepSeekV3DenseBlockParameters& parameters,
    const float* cosine,
    const float* sine,
    const DeepSeekV3DenseBlockConfig& config);

}  // namespace dscuda
