#pragma once

namespace dscuda {

void residual_forward_cpu(
    float* output,
    const float* input,
    const float* branch,
    int elements);

void residual_backward_cpu(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements);

}  // namespace dscuda
