#include "common.cuh"

namespace dscuda {

void mla_compressed_attention_forward_cuda(
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
    cudaStream_t stream) {
    mla_compressed_attention_forward_sm89_cuda(
        output,
        logsumexp,
        query_latent,
        query_rope,
        kv_latent,
        key_rope,
        batch_size,
        sequence_length,
        heads,
        kv_rank,
        rope_size,
        scale,
        stream);
}

void mla_compressed_attention_backward_cuda(
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
    cudaStream_t stream) {
    mla_compressed_attention_backward_sm89_cuda(
        query_latent_gradient,
        query_rope_gradient,
        kv_latent_gradient,
        key_rope_gradient,
        output_gradient,
        output,
        logsumexp,
        query_latent,
        query_rope,
        kv_latent,
        key_rope,
        batch_size,
        sequence_length,
        heads,
        kv_rank,
        rope_size,
        scale,
        stream);
}

std::size_t mla_decode_workspace_elements(
    int batch_size,
    int heads,
    int splits,
    int kv_rank) {
    return mla_decode_workspace_elements_sm89(
        batch_size, heads, splits, kv_rank);
}

void mla_decode_forward_cuda(
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
    cudaStream_t stream) {
    mla_decode_forward_sm89_cuda(
        output,
        logsumexp,
        query,
        paged_kv_cache,
        block_table,
        cache_lengths,
        workspace,
        batch_size,
        heads,
        kv_rank,
        rope_size,
        page_size,
        pages_per_sequence,
        splits,
        scale,
        stream);
}

}  // namespace dscuda
