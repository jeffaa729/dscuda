#pragma once

#include "mla.h"

namespace dscuda {

void mla_compressed_attention_forward_sm89_cuda(
    __nv_bfloat16* output,
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
    cudaStream_t stream);

void mla_compressed_attention_backward_sm89_cuda(
    __nv_bfloat16* query_latent_gradient,
    __nv_bfloat16* query_rope_gradient,
    __nv_bfloat16* kv_latent_gradient,
    __nv_bfloat16* key_rope_gradient,
    const __nv_bfloat16* output_gradient,
    const __nv_bfloat16* output,
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
    cudaStream_t stream);

std::size_t mla_decode_workspace_elements_sm89(
    int batch_size,
    int heads,
    int splits,
    int kv_rank);

void mla_decode_forward_sm89_cuda(
    __nv_bfloat16* output,
    float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* paged_kv_cache,
    const int* block_table,
    const int* cache_lengths,
    float* workspace,
    int batch_size,
    int heads,
    int kv_rank,
    int rope_size,
    int page_size,
    int pages_per_sequence,
    int splits,
    float scale,
    cudaStream_t stream);

}  // namespace dscuda
