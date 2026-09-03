#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Row-major C[M,N] = A[M,K] * B[K,N].
void gemm_fp32_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Same NN layout with BF16 inputs/output and FP32 accumulation.
// Uses Tensor Cores; M, N, and K must be positive multiples of 16.
void gemm_bf16_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
