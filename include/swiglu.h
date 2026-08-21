#pragma once

#include <cuda_runtime.h>

namespace dscuda {

void swiglu_forward_cuda(
    float* output,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream = nullptr);

void swiglu_backward_cuda(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
