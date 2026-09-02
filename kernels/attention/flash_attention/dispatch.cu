#include "common.cuh"

namespace dscuda {

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
    cudaStream_t stream) {
    flash_attention_forward_sm89_cuda(
        output,
        logsumexp,
        query,
        key,
        value,
        batch_size,
        sequence_length,
        heads,
        head_size,
        scale,
        stream);
}

void flash_attention_backward_cuda(
    __nv_bfloat16* query_gradient,
    __nv_bfloat16* key_gradient,
    __nv_bfloat16* value_gradient,
    const __nv_bfloat16* output_gradient,
    const __nv_bfloat16* output,
    const float* logsumexp,
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale,
    cudaStream_t stream) {
    flash_attention_backward_sm89_cuda(
        query_gradient,
        key_gradient,
        value_gradient,
        output_gradient,
        output,
        logsumexp,
        query,
        key,
        value,
        batch_size,
        sequence_length,
        heads,
        head_size,
        scale,
        stream);
}

}  // namespace dscuda
