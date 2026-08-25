#pragma once

#include <cuda_runtime.h>

namespace dscuda {

// Computes the mean negative log-likelihood over row-major [rows, vocabulary]
// logits and saves one log-sum-exp value per row for backward recomputation.
void cross_entropy_forward_cuda(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size,
    cudaStream_t stream = nullptr);

// Overwrites logits_gradient with the gradient of the mean loss. Targets must
// be valid vocabulary IDs, and no probability matrix is materialized.
void cross_entropy_backward_cuda(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
