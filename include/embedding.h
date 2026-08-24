#pragma once

#include <cuda_runtime.h>

namespace dscuda {

// Looks up token_count rows from a row-major [vocabulary_size, hidden_size]
// table. Token IDs must be valid and hidden_size must be divisible by four.
void embedding_forward_cuda(
    float* output,
    const int* token_ids,
    const float* weight,
    int token_count,
    int hidden_size,
    cudaStream_t stream = nullptr);

// Accumulates output_gradient into the selected embedding-table rows. Repeated
// token IDs are combined with atomic additions.
void embedding_backward_cuda(
    float* weight_gradient,
    const float* output_gradient,
    const int* token_ids,
    int token_count,
    int hidden_size,
    cudaStream_t stream = nullptr);

}  // namespace dscuda
