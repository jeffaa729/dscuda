#pragma once

namespace dscuda {

void causal_softmax_forward_cpu(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale);

void causal_softmax_backward_cpu(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale);

}  // namespace dscuda
