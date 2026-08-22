#pragma once

#include <cuda_runtime.h>

namespace dscuda {

void causal_softmax_forward_cuda(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream = nullptr);

// Accumulates the scaled softmax gradient into logits_gradient.
void causal_softmax_backward_cuda(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
