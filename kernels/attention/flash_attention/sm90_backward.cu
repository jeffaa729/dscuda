// Defines the SM90 FlashAttention backward entry point while a Hopper-specific backward kernel is still pending.
// It deliberately reuses the validated SM89 backward implementation so dispatch and file ownership remain explicit.

#include "common.cuh"

namespace dscuda {

void flash_attention_backward_sm90_cuda(__nv_bfloat16* query_gradient, __nv_bfloat16* key_gradient, __nv_bfloat16* value_gradient,
                                        float* query_gradient_accumulator, float* row_delta,
                                        const __nv_bfloat16* output_gradient, const __nv_bfloat16* output, const float* logsumexp,
                                        const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value, int batch_size,
                                        int sequence_length, int heads, int head_size, float scale, cudaStream_t stream) {
    flash_attention_backward_sm89_cuda(query_gradient, key_gradient, value_gradient, query_gradient_accumulator, row_delta, output_gradient, output,
                                       logsumexp, query, key, value, batch_size, sequence_length, heads, head_size, scale, stream);
}

}  // namespace dscuda
