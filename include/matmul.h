#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Computes a strided batch of logical matrix products. Transposed operands are
// stored as [K, M] for the left matrix or [N, K] for the right matrix.
void matmul_fp32_strided_batched_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    int batch_count,
    int left_batch_stride,
    int right_batch_stride,
    int output_batch_stride,
    bool transpose_left,
    bool transpose_right,
    bool accumulate,
    cudaStream_t stream = nullptr);

// Computes output[M, N] = left[M, K] * right[K, N].
void matmul_fp32_forward_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Accumulates dleft = doutput * right^T and dright = left^T * doutput.
void matmul_fp32_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const float* output_gradient,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void matmul_fp32_left_backward_cuda(
    float* left_gradient,
    const float* output_gradient,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void matmul_fp32_right_backward_cuda(
    float* right_gradient,
    const float* left,
    const float* output_gradient,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Multiplies native BF16 operands on Tensor Cores and writes FP32 output; M, N, and K are multiples of 16.
void matmul_bf16_forward_cuda(
    float* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Multiplies native BF16 operands while accumulating both gradients in FP32; M, N, and K are multiples of 16.
void matmul_bf16_backward_cuda(
    float* left_gradient,
    float* right_gradient,
    const __nv_bfloat16* output_gradient,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void matmul_bf16_left_backward_cuda(
    float* left_gradient,
    const __nv_bfloat16* output_gradient,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

void matmul_bf16_right_backward_cuda(
    float* right_gradient,
    const __nv_bfloat16* left,
    const __nv_bfloat16* output_gradient,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
