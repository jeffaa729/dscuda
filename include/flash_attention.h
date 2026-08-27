#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Fuses causal QK^T, online softmax, and PV without materializing the T x T
// score or probability matrices. Q, K, V, output, and gradients use
// [B,T,H,D], while logsumexp uses [B,H,T].
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

}  // namespace dscuda
