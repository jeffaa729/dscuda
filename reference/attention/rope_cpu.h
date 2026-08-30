#pragma once

namespace dscuda {

void rope_forward_cpu(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size);

void rope_backward_cpu(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size);

}  // namespace dscuda
