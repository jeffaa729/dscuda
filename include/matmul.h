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
// SM89 requires M and N multiples of 128 and K a multiple of 32.
// SM90 requires M, N, and K multiples of 64.
void gemm_bf16_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
