#pragma once

namespace dscuda {

void swiglu_forward_cpu(
    float* output,
    const float* gate,
    const float* up,
    int elements);

void swiglu_backward_cpu(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements);

}  // namespace dscuda
