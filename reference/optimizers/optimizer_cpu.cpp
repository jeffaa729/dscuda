// Implements bias-corrected AdamW parameter and moment updates on the CPU.
// Weight decay is applied directly to parameters rather than being mixed into the adaptive gradient moments.

#include "optimizer_cpu.h"

#include <cmath>

namespace dscuda {

void adamw_step_cpu(
    float* parameters,
    float* first_moment,
    float* second_moment,
    const float* gradients,
    int elements,
    int step,
    const AdamWConfig& config) {
    const float first_correction =
        1.0F / (1.0F - std::pow(config.beta1, step));
    const float second_correction =
        1.0F / (1.0F - std::pow(config.beta2, step));

    for (int index = 0; index < elements; ++index) {
        const float gradient = gradients[index];
        const float first =
            config.beta1 * first_moment[index] +
            (1.0F - config.beta1) * gradient;
        const float second =
            config.beta2 * second_moment[index] +
            (1.0F - config.beta2) * gradient * gradient;
        float parameter = parameters[index];
        parameter -=
            config.learning_rate * config.weight_decay * parameter;
        parameter -= config.learning_rate *
            (first * first_correction) /
            (std::sqrt(second * second_correction) + config.epsilon);

        parameters[index] = parameter;
        first_moment[index] = first;
        second_moment[index] = second;
    }
}

}  // namespace dscuda
