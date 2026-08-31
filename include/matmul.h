#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Row-major C[M,N] = op(A)[M,K] * op(B)[K,N], or C += op(A) * op(B).
// Store A as [K,M] when transpose_left is true, and B as [N,K] when transpose_right is true.
void gemm_fp32_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    bool transpose_left = false,
    bool transpose_right = false,
    bool accumulate = false,
    cudaStream_t stream = nullptr);

// Same layout and accumulation contract, with BF16 inputs and FP32 output.
// Uses Tensor Cores; M, N, and K must be positive multiples of 16.
void gemm_bf16_cuda(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    bool transpose_left = false,
    bool transpose_right = false,
    bool accumulate = false,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
