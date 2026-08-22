#pragma once

#include <cuda_runtime.h>

namespace dscuda {

void rope_forward_cuda(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size,
    cudaStream_t stream = nullptr);

// Accumulates the inverse-rotated output gradient into input_gradient.
void rope_backward_cuda(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
