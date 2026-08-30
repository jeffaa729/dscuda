#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace dscuda {

// Implements the absorbed-query form of MLA. Query latents use [B,T,H,C],
// query RoPE uses [B,T,H,R], shared KV latents use [B,T,C], shared key RoPE
// uses [B,T,R], output uses [B,T,H,C], and logsumexp uses [B,H,T].
void mla_compressed_attention_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    cudaStream_t stream = nullptr);

// Recomputes causal probabilities from the saved logsumexp. Every gradient
// element has one writer, and all four FP32 gradient buffers are accumulated.
void mla_compressed_attention_backward_cuda(
    float* query_latent_gradient,
    float* query_rope_gradient,
    float* kv_latent_gradient,
    float* key_rope_gradient,
    const float* output_gradient,
    const float* output,
    const float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_latent,
    const __nv_bfloat16* key_rope,
    int batch_size,
    int sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    float scale,
    cudaStream_t stream = nullptr);

// Returns the FP32 workspace for split maxima, normalizers, and unnormalized
// latent outputs stored as [B,H,S,2+C].
std::size_t mla_decode_workspace_elements(
    int batch_size,
    int heads,
    int splits,
    int kv_rank);

// Decodes one query per batch from a BF16 compressed KV cache. Independent
// split CTAs produce online-softmax states, then one combine CTA merges them.
void mla_decode_forward_cuda(
    float* output,
    float* logsumexp,
    const __nv_bfloat16* query_latent,
    const __nv_bfloat16* query_rope,
    const __nv_bfloat16* kv_cache,
    const __nv_bfloat16* key_rope_cache,
    const int* cache_lengths,
    float* workspace,
    int batch_size,
    int maximum_sequence_length,
    int heads,
    int kv_rank,
    int rope_size,
    int splits,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
