// Implements numerically stable mean vocabulary cross-entropy on the CPU without storing probabilities.
// Backward reconstructs each probability from the saved row log-sum-exp and subtracts the target indicator.

#include "cross_entropy_cpu.h"

#include <algorithm>
#include <cmath>

namespace dscuda {

void cross_entropy_forward_cpu(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size) {
    float loss = 0.0F;
    for (int row = 0; row < rows; ++row) {
        const int offset = row * vocabulary_size;
        float maximum = logits[offset];
        for (int column = 1; column < vocabulary_size; ++column) {
            maximum = std::max(maximum, logits[offset + column]);
        }

        float sum = 0.0F;
        for (int column = 0; column < vocabulary_size; ++column) {
            sum += std::exp(logits[offset + column] - maximum);
        }
        const float row_logsumexp = maximum + std::log(sum);
        logsumexp[row] = row_logsumexp;
        loss += row_logsumexp - logits[offset + targets[row]];
    }
    *mean_loss = loss / rows;
}

void cross_entropy_backward_cpu(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size) {
    const float scale = 1.0F / rows;
    for (int row = 0; row < rows; ++row) {
        const int offset = row * vocabulary_size;
        for (int column = 0; column < vocabulary_size; ++column) {
            const float probability =
                std::exp(logits[offset + column] - logsumexp[row]);
            logits_gradient[offset + column] = scale *
                (probability - static_cast<float>(column == targets[row]));
        }
    }
}

}  // namespace dscuda
