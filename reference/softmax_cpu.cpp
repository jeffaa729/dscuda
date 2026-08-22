// Implements numerically stable scaled causal softmax and its analytical gradient on the CPU.
// Masked keys receive zero probability and contribute no gradient to the original attention logits.

#include "softmax_cpu.h"

#include <algorithm>
#include <cmath>

namespace dscuda {

void causal_softmax_forward_cpu(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float maximum = scale * logits[row_offset];
                for (int key = 1; key <= query; ++key) {
                    maximum = std::max(maximum, scale * logits[row_offset + key]);
                }

                float denominator = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    const float exponential =
                        std::exp(scale * logits[row_offset + key] - maximum);
                    probabilities[row_offset + key] = exponential;
                    denominator += exponential;
                }

                for (int key = 0; key <= query; ++key) {
                    probabilities[row_offset + key] /= denominator;
                }
                for (int key = query + 1; key < sequence_length; ++key) {
                    probabilities[row_offset + key] = 0.0F;
                }
            }
        }
    }
}

void causal_softmax_backward_cpu(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float projection = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    projection += probabilities_gradient[row_offset + key] *
                                  probabilities[row_offset + key];
                }

                for (int key = 0; key <= query; ++key) {
                    const int index = row_offset + key;
                    logits_gradient[index] +=
                        scale * probabilities[index] *
                        (probabilities_gradient[index] - projection);
                }
            }
        }
    }
}

}  // namespace dscuda
