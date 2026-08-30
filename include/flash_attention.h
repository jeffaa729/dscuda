#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Fuses causal QK^T, online softmax, and PV without materializing the T x T
// score or probability matrices. Q, K, V, output, and gradients use
// [B,T,H,D], while logsumexp uses [B,H,T]. Only D=128 is supported.
// BF16 uses Tensor Cores when T is a multiple of 64; FP32 and partial
// sequence tiles use the D128 CUDA-core path. Outputs/gradients remain FP32.
void flash_attention_forward_cuda(
    float* output,
    float* logsumexp,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

void flash_attention_forward_bf16_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

// Recomputes each causal probability from Q, K, and saved logsumexp. The two
// launches partition dQ by query rows and dK/dV by key rows, so every gradient
// element has one writer and the caller's existing values are accumulated.
void flash_attention_backward_cuda(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

void flash_attention_backward_bf16_cuda(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

// Native BF16 IO for same-contract library comparisons: Q/K/V/O/dO/dQ/dK/dV
// are BF16, accumulators and LSE are FP32. Requires D=128 and T a multiple of 64.
// Unlike the FP32-output functions above, backward OVERWRITES gradients.
void flash_attention_forward_bf16_io_cuda(
    __nv_bfloat16* output, float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, int head_size,
    float scale, cudaStream_t stream = nullptr);

void flash_attention_backward_bf16_io_cuda(
    __nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, int head_size,
    float scale, cudaStream_t stream = nullptr);

}  // namespace dscuda
