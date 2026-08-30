// Implements residual addition and its two identity-gradient branches on the CPU.
// The backward reference accumulates because both destinations may receive gradients elsewhere in the graph.

#include "residual_cpu.h"

namespace dscuda {

void residual_forward_cpu(
    float* output,
    const float* input,
    const float* branch,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        output[index] = input[index] + branch[index];
    }
}

void residual_backward_cpu(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        input_gradient[index] += output_gradient[index];
        branch_gradient[index] += output_gradient[index];
    }
}

}  // namespace dscuda
