// Implements token-table lookup and accumulated embedding gradients on the CPU.
// The scalar reference intentionally exposes repeated-token accumulation without layout or vectorization optimizations.

#include "embedding_cpu.h"

namespace dscuda {

void embedding_forward_cpu(
    float* output,
    const int* token_ids,
    const float* weight,
    int token_count,
    int hidden_size) {
    for (int token = 0; token < token_count; ++token) {
        const int source = token_ids[token] * hidden_size;
        const int destination = token * hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            output[destination + column] = weight[source + column];
        }
    }
}

void embedding_backward_cpu(
    float* weight_gradient,
    const float* output_gradient,
    const int* token_ids,
    int token_count,
    int hidden_size) {
    for (int token = 0; token < token_count; ++token) {
        const int destination = token_ids[token] * hidden_size;
        const int source = token * hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            weight_gradient[destination + column] +=
                output_gradient[source + column];
        }
    }
}

}  // namespace dscuda
