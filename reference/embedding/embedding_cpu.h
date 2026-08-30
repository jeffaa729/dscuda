#pragma once

namespace dscuda {

void embedding_forward_cpu(
    float* output,
    const int* token_ids,
    const float* weight,
    int token_count,
    int hidden_size);

// Accumulates gradients so repeated tokens and tied output-head gradients can
// share the same embedding-table gradient buffer.
void embedding_backward_cpu(
    float* weight_gradient,
    const float* output_gradient,
    const int* token_ids,
    int token_count,
    int hidden_size);

}  // namespace dscuda
