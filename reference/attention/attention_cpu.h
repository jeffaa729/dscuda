#pragma once

namespace dscuda {

void dense_attention_forward_cpu(
    float* output,
    float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale);

void dense_attention_backward_cpu(
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    const float* output_gradient,
    const float* probabilities,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    float scale);

}  // namespace dscuda
