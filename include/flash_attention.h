#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Fuses causal QK^T, online softmax, and PV without materializing the T x T
// score or probability matrices. Q/O use BF16 [B,T,Hq,128], K/V use BF16
// [B,T,Hkv,128], and the natural-log LSE uses FP32 [B,Hq,T]. Hq must be
// divisible by Hkv, and T must be a positive multiple of 64.
void flash_attention_forward_cuda(
    __nv_bfloat16* output,
    float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int sequence_length,
    int query_heads,
    int key_value_heads,
    int head_size,
    float scale,
    cudaStream_t stream = nullptr);

// Recomputes probabilities from Q, K, and LSE. dO/dQ/dK/dV use BF16.
// dq_accumulator is FP32 [B,T,H,128] and row_delta is FP32 [B,H,T];
// callers own both workspaces so the complete pipeline is CUDA-Graph-safe.
void flash_attention_backward_cuda(
    __nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    float* query_gradient_accumulator, float* row_delta,
    const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, int head_size,
    float scale, cudaStream_t stream = nullptr);

}  // namespace dscuda
