#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dscuda {

// Fuses causal QK^T, online softmax, and PV without materializing the T x T
// score or probability matrices. Q/K/V/O use BF16 [B,T,H,128], while the
// natural-log LSE uses FP32 [B,H,T]; T must be a positive multiple of 64.
void flash_attention_forward_cuda(
    __nv_bfloat16* output,
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

// Recomputes probabilities from Q, K, and LSE. dO/dQ/dK/dV use BF16 and each
// gradient element has one writer, so the output gradient buffers are replaced.
void flash_attention_backward_cuda(
    __nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
    const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
    const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
    int batch_size, int sequence_length, int heads, int head_size,
    float scale, cudaStream_t stream = nullptr);

}  // namespace dscuda
