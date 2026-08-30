// Computes expert-grouped linear forward and backward with scalar CPU loops.
// It provides the correctness reference for both CUDA GEMM precision paths.

#include "grouped_gemm_cpu.h"

#include <algorithm>

namespace dscuda {

void grouped_linear_forward_cpu(
    float* output,
    const float* input,
    const float* weight,
    const int* slot_expert,
    int dispatched_rows,
    int output_size,
    int input_size) {
    for (int row = 0; row < dispatched_rows; ++row) {
        const int expert = slot_expert[row];
        for (int column = 0; column < output_size; ++column) {
            float sum = 0.0F;
            for (int inner = 0; inner < input_size; ++inner) {
                sum += input[row * input_size + inner] *
                       weight[(expert * input_size + inner) * output_size
                              + column];
            }
            output[row * output_size + column] = sum;
        }
    }
}

void grouped_linear_backward_cpu(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const int* expert_offsets,
    const int* slot_expert,
    int dispatched_rows,
    int experts,
    int output_size,
    int input_size,
    bool accumulate_input) {
    if (!accumulate_input) {
        std::fill_n(input_gradient, dispatched_rows * input_size, 0.0F);
    }
    for (int row = 0; row < dispatched_rows; ++row) {
        const int expert = slot_expert[row];
        for (int inner = 0; inner < input_size; ++inner) {
            for (int column = 0; column < output_size; ++column) {
                input_gradient[row * input_size + inner] +=
                    output_gradient[row * output_size + column] *
                    weight[(expert * input_size + inner) * output_size
                           + column];
            }
        }
    }
    for (int expert = 0; expert < experts; ++expert) {
        for (int inner = 0; inner < input_size; ++inner) {
            for (int column = 0; column < output_size; ++column) {
                float sum = 0.0F;
                for (int row = expert_offsets[expert];
                     row < expert_offsets[expert + 1];
                     ++row) {
                    sum += input[row * input_size + inner] *
                           output_gradient[row * output_size + column];
                }
                weight_gradient[
                    (expert * input_size + inner) * output_size + column] +=
                    sum;
            }
        }
    }
}

}  // namespace dscuda
