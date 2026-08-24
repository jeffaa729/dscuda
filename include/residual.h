#pragma once

#include <cuda_runtime.h>

namespace dscuda {

// All buffers contain a multiple of four FP32 elements.
void residual_forward_cuda(
    float* output,
    const float* input,
    const float* branch,
    int elements,
    cudaStream_t stream = nullptr);

// Adds output_gradient into both input-gradient buffers.
void residual_backward_cuda(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
