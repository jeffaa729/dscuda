#pragma once

#include "optimizer.h"

namespace dscuda {

void adamw_step_cpu(
    float* parameters,
    float* first_moment,
    float* second_moment,
    const float* gradients,
    int elements,
    int step,
    const AdamWConfig& config);

}  // namespace dscuda
