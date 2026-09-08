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

// BF16 inputs/output and FP32 accumulation; layout depends on the GPU.
// SM89: row-major C[M,N] = A[M,K] * B[K,N]; M,N multiples of 128, K of 32.
// SM90: row-major A[M,K], column-major B[K,N] (physical [N,K]),
// and column-major C[M,N], matching fast.cu; M,N multiples of 2048, K of 64.
void gemm_bf16_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
