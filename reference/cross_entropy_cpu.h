#pragma once

namespace dscuda {

void cross_entropy_forward_cpu(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size);

void cross_entropy_backward_cpu(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size);

}  // namespace dscuda
