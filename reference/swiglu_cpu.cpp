// Implements a readable FP32 SwiGLU forward and backward reference for numerical verification.
// The scalar loop mirrors the CUDA equations without introducing parallel reduction or accumulation differences.

#include "swiglu_cpu.h"

#include <cmath>

namespace dscuda {
namespace {

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

}  // namespace

void swiglu_forward_cpu(
    float* output,
    const float* gate,
    const float* up,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        output[index] = gate_value * sigmoid(gate_value) * up[index];
    }
}

void swiglu_backward_cpu(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid_value = sigmoid(gate_value);
        const float silu_value = gate_value * sigmoid_value;
        const float silu_gradient =
            sigmoid_value * (1.0F + gate_value * (1.0F - sigmoid_value));

        gate_gradient[index] = output_gradient[index] * up[index] * silu_gradient;
        up_gradient[index] = output_gradient[index] * silu_value;
    }
}

}  // namespace dscuda
