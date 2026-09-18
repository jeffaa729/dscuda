#include "common.cuh"
#include "cuda_common.h"

namespace dscuda {
namespace {

bool use_sm90() {
    int device = 0;
    int major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    return major == 9;
}

}  // namespace

void flash_attention_forward_cuda(__nv_bfloat16* output, float* logsumexp, const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
                                  int batch_size, int sequence_length, int query_heads, int key_value_heads, int head_size, float scale,
                                  cudaStream_t stream) {
    if (use_sm90()) {
        flash_attention_forward_sm90_cuda(output, logsumexp, query, key, value, batch_size, sequence_length, query_heads, key_value_heads, head_size,
                                          scale, stream);
    } else {
        flash_attention_forward_sm89_cuda(output, logsumexp, query, key, value, batch_size, sequence_length, query_heads, key_value_heads, head_size,
                                          scale, stream);
    }
}

void flash_attention_backward_cuda(__nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
                                   float* query_gradient_accumulator, float* row_delta,
                                   const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp, const __nv_bfloat16* query,
                                   const __nv_bfloat16* key, const __nv_bfloat16* value, int batch_size, int sequence_length, int heads, int head_size,
                                   float scale, cudaStream_t stream) {
    if (use_sm90()) {
        flash_attention_backward_sm90_cuda(query_gradient, key_gradient, value_gradient, query_gradient_accumulator, row_delta, output_gradient, output,
                                           logsumexp, query, key, value, batch_size, sequence_length, heads, head_size, scale, stream);
    } else {
        flash_attention_backward_sm89_cuda(query_gradient, key_gradient, value_gradient, query_gradient_accumulator, row_delta, output_gradient, output,
                                           logsumexp, query, key, value, batch_size, sequence_length, heads, head_size, scale, stream);
    }
}

}  // namespace dscuda
