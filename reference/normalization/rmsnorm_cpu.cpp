// Implements a transparent FP32 CPU reference for RMSNorm forward and manually derived backward calculations.
// It prioritizes readable mathematics and deterministic accumulation order over performance so CUDA kernels have a trustworthy oracle.

#include "rmsnorm_cpu.h"

#include <cmath>

namespace dscuda {

void rmsnorm_forward_cpu(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows,
    int hidden_size,
    float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const int offset = row * hidden_size;
        float sum_of_squares = 0.0F;
        for (int column = 0; column < hidden_size; ++column) {
            const float value = input[offset + column];
            sum_of_squares += value * value;
        }

        const float scale = 1.0F / std::sqrt(sum_of_squares / hidden_size + epsilon);
        inverse_rms[row] = scale;
        for (int column = 0; column < hidden_size; ++column) {
            output[offset + column] = input[offset + column] * scale * weight[column];
        }
    }
}

void rmsnorm_backward_cpu(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows,
    int hidden_size) {
    for (int row = 0; row < rows; ++row) {
        const int offset = row * hidden_size;
        const float scale = inverse_rms[row];
        float projection = 0.0F;

        for (int column = 0; column < hidden_size; ++column) {
            projection += output_gradient[offset + column] * weight[column] * input[offset + column];
        }

        const float correction = projection * scale * scale * scale / hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            const int index = offset + column;
            const float upstream_times_weight = output_gradient[index] * weight[column];
            input_gradient[index] += scale * upstream_times_weight - input[index] * correction;
            weight_gradient[column] += output_gradient[index] * input[index] * scale;
        }
    }
}

}  // namespace dscuda
