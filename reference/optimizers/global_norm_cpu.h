#pragma once

namespace dscuda {

float global_norm_cpu(const float* gradients, int elements);

void clip_gradients_cpu(
    float* gradients,
    int elements,
    float norm,
    float max_norm);

}  // namespace dscuda
