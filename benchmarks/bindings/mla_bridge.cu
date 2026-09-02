// Exposes standalone compressed MLA forward, backward, and decode to Python-owned CUDA buffers.
// Activations and gradients use BF16 storage; natural-log LSE and workspace use FP32.
#include "mla.h"

#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char* dscuda_mla_last_error() {
    return last_error.c_str();
}

extern "C" int dscuda_mla_forward(
    __nv_bfloat16* output, float* lse,
    const __nv_bfloat16* query, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* latent, const __nv_bfloat16* key_rope,
    int batch, int sequence, int heads, int rank, int rope, float scale,
    cudaStream_t stream) {
    try {
        dscuda::mla_compressed_attention_forward_cuda(
            output, lse, query, query_rope, latent, key_rope,
            batch, sequence, heads, rank, rope, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" int dscuda_mla_backward(
    __nv_bfloat16* dquery, __nv_bfloat16* dquery_rope,
    __nv_bfloat16* dlatent, __nv_bfloat16* dkey_rope,
    const __nv_bfloat16* dout, const __nv_bfloat16* output, const float* lse,
    const __nv_bfloat16* query, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* latent, const __nv_bfloat16* key_rope,
    int batch, int sequence, int heads, int rank, int rope, float scale,
    cudaStream_t stream) {
    try {
        dscuda::mla_compressed_attention_backward_cuda(
            dquery, dquery_rope, dlatent, dkey_rope, dout, output, lse,
            query, query_rope, latent, key_rope,
            batch, sequence, heads, rank, rope, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" std::size_t dscuda_mla_workspace_elements(
    int batch, int heads, int splits, int rank) {
    return dscuda::mla_decode_workspace_elements(batch, heads, splits, rank);
}

extern "C" int dscuda_mla_decode(
    __nv_bfloat16* output, float* lse,
    const __nv_bfloat16* query, const __nv_bfloat16* paged_cache,
    const int* block_table, const int* lengths, float* workspace,
    int batch, int heads, int rank, int rope, int page_size, int pages, int splits,
    float scale, cudaStream_t stream) {
    try {
        dscuda::mla_decode_forward_cuda(
            output, lse, query, paged_cache, block_table, lengths, workspace,
            batch, heads, rank, rope, page_size, pages, splits, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
