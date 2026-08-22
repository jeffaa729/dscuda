#pragma once

#include <cuda_runtime.h>

namespace dscuda {

// Computes output[M, N] = left[M, K] * right[K, N].
void matmul_forward_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Accumulates dleft = doutput * right^T and dright = left^T * doutput.
void matmul_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
