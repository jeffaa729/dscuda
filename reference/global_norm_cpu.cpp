// Computes the L2 norm of a flattened gradient buffer and clips it in place on the CPU.
// Double-precision reference accumulation limits oracle error while the resulting norm and scale remain FP32.

#include "global_norm_cpu.h"

#include <algorithm>
#include <cmath>

namespace dscuda {

float global_norm_cpu(const float* gradients, int elements) {
    double sum = 0.0;
    for (int index = 0; index < elements; ++index) {
        const double gradient = gradients[index];
        sum += gradient * gradient;
    }
    return static_cast<float>(std::sqrt(sum));
}

void clip_gradients_cpu(
    float* gradients,
    int elements,
    float norm,
    float max_norm) {
    const float scale = std::min(1.0F, max_norm / norm);
    for (int index = 0; index < elements; ++index) {
        gradients[index] *= scale;
    }
}

}  // namespace dscuda
