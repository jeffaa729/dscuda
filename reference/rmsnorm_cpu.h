#pragma once

namespace dscuda {

void rmsnorm_forward_cpu(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows,
    int hidden_size,
    float epsilon);

// Accumulates into both gradient buffers so the reference follows the same
// computation-graph semantics as the CUDA implementation.
void rmsnorm_backward_cpu(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows,
    int hidden_size);

}  // namespace dscuda
