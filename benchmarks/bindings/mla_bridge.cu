// Exposes standalone compressed MLA forward, backward, and decode to Python-owned CUDA buffers.
// All operands use BF16 storage; outputs, natural-log LSE, and gradients use FP32.
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
    float* output, float* lse,
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
    float* dquery, float* dquery_rope, float* dlatent, float* dkey_rope,
    const float* dout, const float* output, const float* lse,
    const __nv_bfloat16* query, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* latent, const __nv_bfloat16* key_rope,
    int batch, int sequence, int heads, int rank, int rope, float scale,
    int accumulate, cudaStream_t stream) {
    try {
        dscuda::mla_compressed_attention_backward_cuda(
            dquery, dquery_rope, dlatent, dkey_rope, dout, output, lse,
            query, query_rope, latent, key_rope,
            batch, sequence, heads, rank, rope, scale, accumulate, stream);
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
    float* output, float* lse,
    const __nv_bfloat16* query, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* latent, const __nv_bfloat16* key_rope,
    const int* lengths, float* workspace,
    int batch, int sequence, int heads, int rank, int rope, int splits,
    float scale, cudaStream_t stream) {
    try {
        dscuda::mla_decode_forward_cuda(
            output, lse, query, query_rope, latent, key_rope, lengths, workspace,
            batch, sequence, heads, rank, rope, splits, scale, stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
